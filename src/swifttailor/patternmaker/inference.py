# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""PatternMaker Stage 1: single text/image input -> sewing pattern.

Self-contained (no DeepSpeed / DDP) single-input inference, factored out of
``scripts/run_patternmaker.py`` (the DeepSpeed eval harness) and the generation
body of ``patternmaker.trainer.generation_step``. Used by ``scripts/run_e2e.py``
to produce a ``*_specification.json`` from an arbitrary text prompt and/or image
file, which the GarmentSewer stage then reconstructs into a 3D mesh.

The heavy lifting is reused as-is:
  * ``InferenceDataset`` turns a ``{type, inputs}`` datapoint into the 9-tuple
    that ``collate_fn`` consumes.
  * ``collate_fn(generation_only=True)`` emits exactly the tensors ``model.chat``
    needs (no ground-truth pattern required).
  * ``dataset.decode`` turns the model output back into an ``NNSewingPattern``.
"""

import json
import os
import re
import tempfile
from functools import partial
from typing import List, Optional

import torch
import transformers
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from swifttailor.config import GLOBAL_CONFIG
from swifttailor.data.data_wrappers.collate_fns import collate_fn
from swifttailor.data.datasets.constants import IMG_CONTEXT_TOKEN
from swifttailor.data.datasets.inference_dataset import InferenceDataset
from swifttailor.patternmaker.st_model import SwiftTailorConfig, SwiftTailorForCausalLM
from swifttailor.patternmaker.trainer_utils import dict_to_cpu, dict_to_cuda, dict_to_dtype

# Model-input keys cast to the compute dtype before generation (mirrors
# trainer.generation_step).
_CAST_KEYS = [
    "pixel_values",
    "questions_pattern_endpoints",
    "questions_pattern_transformations",
]


def extract_model_config(cfg) -> SwiftTailorConfig:
    """Merge the pretrained ``config.json`` with the Hydra ``model`` overrides.

    Shared by ``scripts/run_patternmaker.py`` and ``scripts/run_e2e.py``.
    """
    if cfg.pretrained_config is None:
        if cfg.my_pretrained_path is not None:
            cfg.pretrained_config = cfg.my_pretrained_path + "/config.json"
        elif cfg.llm_pretrained_path is not None:
            cfg.pretrained_config = cfg.llm_pretrained_path + "/config.json"

    assert os.path.exists(
        cfg.pretrained_config
    ), f"Pretrained config file {cfg.pretrained_config} does not exist."
    additional_model_config = OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)

    with open(cfg.pretrained_config, "r") as f:
        pretrained_config = json.load(f)
    model_config = {
        **pretrained_config,
        **additional_model_config,
    }
    return SwiftTailorConfig(**model_config)


def _slug(text: str) -> str:
    """Slugify text into a short filesystem-safe identifier."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s[:40] or "text_prompt"


def build_inference_dataset(
    *,
    text: Optional[str],
    image: Optional[str],
    garment_tokenizer,
    panel_classification: str,
    image_size: int = 448,
    max_num_patches: int = 12,
    conv_template: str = "internvl2_5",
) -> InferenceDataset:
    """Wrap a single text/image request in an ``InferenceDataset``.

    Datapoint type: image only -> ``image``; text only -> ``description``;
    both -> ``text_image``.
    """
    if not text and not image:
        raise ValueError("At least one of `text` or `image` must be provided.")

    if image and text:
        datapoint = {"type": "text_image", "inputs": {"image_path": image, "description": text}}
    elif image:
        datapoint = {"type": "image", "inputs": {"image_path": image}}
    else:
        datapoint = {"type": "description", "inputs": {"description": text}}

    GLOBAL_CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    fd, inference_json = tempfile.mkstemp(
        prefix="st_inference_", suffix=".json", dir=str(GLOBAL_CONFIG.cache_dir)
    )
    with os.fdopen(fd, "w") as f:
        json.dump([datapoint], f)

    return InferenceDataset(
        inference_json=inference_json,
        garment_tokenizer=garment_tokenizer,
        panel_classification=panel_classification,
        image_size=image_size,
        max_num_patches=max_num_patches,
        conv_template=conv_template,
    )


def load_patternmaker(
    *,
    my_pretrained_path: Optional[str],
    llm_pretrained_path: Optional[str],
    model_config: SwiftTailorConfig,
    dataset: InferenceDataset,
    torch_dtype: torch.dtype,
    device: str,
):
    """Build the SwiftTailor VLM + tokenizer for inference (no DeepSpeed).

    Mirrors the model/tokenizer setup in ``scripts/run_patternmaker.py::main``
    but drops the training-only bits (gradient checkpointing, DeepSpeed engine).
    """
    tokenizer_path = llm_pretrained_path or my_pretrained_path
    assert (
        tokenizer_path is not None
    ), "Either llm_pretrained_path or my_pretrained_path must be specified."
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=None,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    all_new_tokens = dataset.get_all_token_names()
    tokenizer.add_tokens(all_new_tokens)
    token_name2_idx = {
        token: tokenizer(token, add_special_tokens=False).input_ids[0] for token in all_new_tokens
    }
    dataset.set_token_indices(token_name2_idx)

    if my_pretrained_path is not None:
        model: SwiftTailorForCausalLM = SwiftTailorForCausalLM.from_pretrained(
            my_pretrained_path,
            config=model_config,
            panel_edge_indices=dataset.panel_edge_type_indices,
            gt_stats=dataset.gt_stats,
            torch_dtype=torch_dtype,
        )
    else:
        model = SwiftTailorForCausalLM(
            config=model_config,
            panel_edge_indices=dataset.panel_edge_type_indices,
            gt_stats=dataset.gt_stats,
            llm_pretrained_path=llm_pretrained_path,
        ).to(torch_dtype)

    model.resize_token_embeddings(len(tokenizer))
    model.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.to(device).eval()
    return model, tokenizer


def _sample_name(data_type: str, inputs: dict, idx: int) -> str:
    """Derive an output sample name from the datapoint's type and inputs."""
    if data_type in ("image", "text_image"):
        return os.path.splitext(os.path.basename(inputs["image_path"]))[0]
    if data_type in ("description", "occasion"):
        return _slug(inputs["description"])
    return f"sample_{idx}"


def generate_specs(
    *,
    model: SwiftTailorForCausalLM,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: InferenceDataset,
    out_dir: str,
    torch_dtype: torch.dtype,
    conv_type: str = "internvl2_5",
) -> List[str]:
    """Run PatternMaker over the dataset, writing one spec per datapoint.

    Returns the list of ``*_specification.json`` paths (one per datapoint that
    decoded to a valid pattern).
    """
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=conv_type,
            num_image_token=model.num_image_token,
            generation_only=True,
        ),
    )
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    spec_paths: List[str] = []
    for idx, input_dict in enumerate(loader):
        data_type, inputs = dataset.get_item_infos(idx)
        name = _sample_name(data_type, inputs, idx)

        input_dict = dict_to_cuda(input_dict)
        input_dict = dict_to_dtype(input_dict, torch_dtype, _CAST_KEYS)

        with torch.no_grad():
            output_dict = model.chat(
                pixel_values=input_dict["pixel_values"],
                input_ids=input_dict["question_ids"],
                attention_mask=input_dict["question_attention_masks"],
                endpoints=input_dict["questions_pattern_endpoints"],
                endpoints_mask=input_dict["questions_pattern_endpoints_mask"],
                transformations=input_dict["questions_pattern_transformations"],
                transformations_mask=input_dict["questions_pattern_transformations_mask"],
                max_new_tokens=2100,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True,
                temperature=1.1,
                top_p=0.85,
                top_k=50,
            )
        output_dict = dict_to_cpu(output_dict)
        output_dict = dict_to_dtype(output_dict, torch.float32)
        output_dict["input_mask"] = (
            torch.arange(output_dict["output_ids"].shape[1]).reshape(1, -1)
            >= input_dict["question_ids"].shape[1]
        )

        output_text, pattern, error_type = dataset.decode(output_dict, tokenizer)

        sample_dir = os.path.join(out_dir, name)
        os.makedirs(sample_dir, exist_ok=True)
        pattern.name = name
        pattern.serialize(
            sample_dir,
            spec_only=False,
            with_3d=False,
            with_text=False,
            view_ids=False,
            to_subfolder=False,
        )
        spec_path = os.path.join(sample_dir, f"{name}_specification.json")
        print(f"[patternmaker] {name}: {spec_path} (decode: {error_type})")
        spec_paths.append(spec_path)

    return spec_paths
