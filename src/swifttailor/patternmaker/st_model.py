# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""SwiftTailor VLM: an InternVL chat model extended to regress sewing-pattern edges."""

import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import transformers
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, Qwen2ForCausalLM
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from swifttailor.data.datasets.panel_configs import StandardizeConfig
from swifttailor.data.garment_tokenizers.special_tokens import PanelEdgeType, PanelEdgeTypeIndices

from .encodings import DiscreteEncoding, SinusoidalEncoding
from .internvl.conversation import get_conv_template
from .internvl.modeling_internvl_chat import InternVisionModel, InternVLChatConfig

try:
    from flash_attn.bert_padding import pad_input, unpad_input  # noqa
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func  # noqa

    has_flash_attn = True  # fix later
    print("FlashAttention2 is installed.")
except Exception:
    print("FlashAttention2 is not installed.")
    has_flash_attn = False


logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op="eq"):
    """Compare two version strings using the named ``operator`` function."""
    import operator

    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def make_mlp(input_dim, hidden_dim, output_dim, num_layers, dropout=0):
    """Very simple multi-layer perceptron (also called FFN)."""
    h = [input_dim] + [hidden_dim] * (num_layers - 1)
    layers = []
    for i in range(num_layers - 1):
        layers.append(nn.Linear(h[i], h[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(h[-1], output_dim))
    layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _discretize(x, bin, bounds, dim):
    """Clamp to ``bounds`` and snap each value to one of 256 uniform bins."""
    min_bounds = torch.as_tensor(bounds[:dim], device=x.device, dtype=x.dtype)
    max_bounds = torch.as_tensor(bounds[dim:], device=x.device, dtype=x.dtype)
    x = torch.minimum(max_bounds, x)
    x = torch.maximum(min_bounds, x)
    x = (x - min_bounds) / (max_bounds - min_bounds)
    x = torch.round(x * 256) / 256
    x = x * (max_bounds - min_bounds) + min_bounds
    return x


class SwiftTailorConfig(InternVLChatConfig):
    """InternVL chat config extended with edge-regression and encoding options."""

    DEFAULT_VERT_BOUNDS = [-4, -4, 4, 4]
    DEFAULT_TRANSF_BOUNDS = [-4, -4, -4, -1, -1, -1, -1, 4, 4, 4, 1, 1, 1, 1]

    def __init__(
        self,
        edge_loss_weight: float = 0.1,
        pos_embed: bool = True,
        num_freq: int = 9,
        pos_embed_type: str = "sinusoidal",
        vert_bounds: List[float] = DEFAULT_VERT_BOUNDS,
        transf_bounds: List[float] = DEFAULT_TRANSF_BOUNDS,
        denormolize_for_loss: bool = False,
        num_regression_layers: int = 2,
        discretize: bool = True,
        bin_num: int = 128,
        **kwargs,
    ):
        """Store edge-regression, positional-encoding, and discretization settings."""
        super().__init__(**kwargs)
        self.edge_loss_weight = edge_loss_weight
        self.pos_embed = pos_embed
        self.num_freq = num_freq
        self.pos_embed_type = pos_embed_type
        self.verts_bounds = vert_bounds
        self.transf_bounds = transf_bounds
        self.denormalize_for_loss = denormolize_for_loss
        self.num_regression_layers = num_regression_layers
        self.discretize = discretize
        self.bin_num = bin_num
        if "llm_config" in kwargs:
            self.vocab_size = kwargs["llm_config"]["vocab_size"]
        else:
            self.vocab_size = None


class SwiftTailorMetaModel:
    """Mixin adding edge/transformation regression heads and positional encodings."""

    def __init__(
        self,
        config: SwiftTailorConfig,
        panel_edge_indices: PanelEdgeTypeIndices,
        **kwargs,
    ):
        """Store config and edge indices, then build the panel-edge modules."""
        super().__init__(config)
        self.config = config

        self.panel_edge_indices = panel_edge_indices

        if len(kwargs) > 0:
            logger.warning(
                f"[WARNING] Unused kwargs in {self.__class__.__name__}: "
                f"{kwargs.keys()}. Please check your configuration."
            )

        self.initialize_panel_edge_modules()

    def initialize_panel_edge_modules(self):
        """Build the regression MLPs and, if enabled, positional encoders/projections.

        Raises:
            ValueError: If ``pos_embed_type`` is not supported.
        """
        in_dim = self.config.hidden_size

        self.transformation_fc = make_mlp(
            in_dim,
            in_dim,
            self.panel_edge_indices.get_index_param_num(self.panel_edge_indices.move_idx),
            self.config.num_regression_layers,
        )
        self.transformation_fc.train()
        for p in self.transformation_fc.parameters():
            p.requires_grad = True

        line_curve_out_dim = (
            4 if self.panel_edge_indices.get_token_indices(PanelEdgeType.CUBIC) == -1 else 6
        )
        if self.panel_edge_indices.get_token_indices(PanelEdgeType.ARC) != -1:
            line_curve_out_dim += 2

        self.line_curve_fc = make_mlp(
            in_dim, in_dim, line_curve_out_dim, self.config.num_regression_layers
        )
        self.line_curve_fc.train()
        for p in self.line_curve_fc.parameters():
            p.requires_grad = True

        if self.config.pos_embed:
            if self.config.pos_embed_type == "sinusoidal":
                self.vertex_encoding = SinusoidalEncoding(
                    in_dim=2,
                    num_frequencies=self.config.num_freq,
                    min_freq_exp=0.0,
                    max_freq_exp=self.config.num_freq - 1,
                    include_input=True,
                )
                self.trasl_encoding = SinusoidalEncoding(
                    in_dim=3,
                    num_frequencies=self.config.num_freq,
                    min_freq_exp=0.0,
                    max_freq_exp=self.config.num_freq - 1,
                    include_input=True,
                )
                self.transf_proj = nn.Sequential(
                    nn.Linear(self.trasl_encoding.get_out_dim() + 4, in_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(in_dim, in_dim),
                )
                self.vertex_proj = nn.Sequential(
                    nn.Linear(self.vertex_encoding.get_out_dim(), in_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(in_dim, in_dim),
                )

            elif self.config.pos_embed_type == "discrete":
                self.vertex_encoding = DiscreteEncoding(
                    in_dim=2,
                    out_dim=in_dim,
                    bin_num=self.config.bin_num,
                    max_bounds=self.config.verts_bounds[2:],
                    min_bounds=self.config.verts_bounds[:2],
                )
                self.trasl_encoding = DiscreteEncoding(
                    in_dim=7,
                    out_dim=in_dim,
                    bin_num=self.config.bin_num,
                    max_bounds=self.config.transf_bounds[7:],
                    min_bounds=self.config.transf_bounds[:7],
                )
                self.transf_proj = nn.Sequential(
                    nn.Linear(self.trasl_encoding.get_out_dim(), in_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(in_dim, in_dim),
                )
                self.vertex_proj = nn.Sequential(
                    nn.Linear(self.vertex_encoding.get_out_dim(), in_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(in_dim, in_dim),
                )

            else:
                raise ValueError(
                    f"Unsupported pos_embed_type: {self.config.pos_embed_type}. Supported types are 'sinusoidal' and 'discrete'."
                )

    def get_trainable_param_names(self):
        """Return the names of the edge-regression (and encoding) submodules to train."""
        l = ["line_curve_fc", "transformation_fc"]
        if self.config.pos_embed:
            l.extend(["vertex_proj", "transf_proj"])
        return l

    def init_trainable_params(self):
        """Zero-initialize the final layers of the regression heads and projections."""
        self.line_curve_fc[-2].weight.data.zero_()
        self.transformation_fc[-2].weight.data.zero_()
        if self.config.pos_embed:
            self.vertex_proj[-1].weight.data.zero_()
            self.transf_proj[-1].weight.data.zero_()


def denormalize(gt_stats: StandardizeConfig, params: torch.FloatTensor, is_transf: bool = False):
    """Undo the training normalization on vertex or transformation parameters."""
    shift = (
        gt_stats.translations.shift + gt_stats.rotations.shift
        if is_transf
        else gt_stats.vertices.shift
    )
    scale = (
        gt_stats.translations.scale + gt_stats.rotations.scale
        if is_transf
        else gt_stats.vertices.scale
    )
    shift = torch.tensor(shift).to(params)
    scale = torch.tensor(scale).to(params)
    if params.ndim == 1:
        params = params.unsqueeze(0)
    assert params.shape[-1] == len(shift) == len(scale)
    params = (params * scale) + shift
    return params


class SwiftTailorModel(SwiftTailorMetaModel, PreTrainedModel):
    """Vision + language backbone that injects image and edge/transform embeddings."""

    config_class = SwiftTailorConfig
    base_model_prefix = "language_model"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternVisionModel", "LlamaDecoderLayer", "Qwen2DecoderLayer"]

    def __init__(
        self,
        config: SwiftTailorConfig,
        vision_model=None,
        language_model=None,
        use_flash_attn=True,
        **kwargs,
    ):
        """Build or attach the vision tower, language model, and projection MLP.

        Raises:
            NotImplementedError: If the configured LLM architecture is unsupported.
        """
        super(SwiftTailorModel, self).__init__(config, **kwargs)

        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        self.vision_model: InternVisionModel  # type hint
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == "Qwen2ForCausalLM":
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(
                    f"{config.llm_config.architectures[0]} is not implemented."
                )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.img_context_token_id = None

    def pixel_shuffle(self, x, scale_factor=0.5):
        """Rearrange spatial resolution into channels by the given scale factor."""
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale
        # --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have "
                "not been swapped back, which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        """Encode images with the vision tower and project them to LLM embeddings."""
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def prepare_forward(
        self,
        *,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_ids: torch.LongTensor = None,
        edge_mask: Optional[torch.BoolTensor] = None,
        transf_mask: Optional[torch.BoolTensor] = None,
        endpoints: Optional[torch.FloatTensor] = None,
        endpoints_mask: Optional[torch.BoolTensor] = None,
        transformations: Optional[torch.FloatTensor] = None,
        transformations_mask: Optional[torch.BoolTensor] = None,
    ):
        """Build input embeddings, splicing in image, endpoint, and transform features.

        Returns:
            The token embeddings with visual and positional information injected.
        """
        # Prepare text inputs
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        assert (
            self.img_context_token_id is not None
        ), "img_context_token_id must be set before training."
        selected = input_ids == self.img_context_token_id
        if selected.sum() > 0:
            # Prepare image inputs
            vit_embeds = self.extract_feature(pixel_values)

            # Make sure that the number of patch tokens equals to
            # that of placeholder.
            # Each sample may contains multiple images (or patches).
            assert (
                (selected.sum() == vit_embeds.size(0) * self.num_image_token)
                # No sample has image, but we still have one placeholder
                or (selected.sum() == 0 and vit_embeds.size(0) == 1)
            ), (
                f"inputs_embeds[selected].shape={inputs_embeds[selected].shape},"
                f"vit_embeds.shape={vit_embeds.shape}, "
                f"selected.sum()={selected.sum()}, "
                f"num_image_token={self.num_image_token}"
            )

            inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, vit_embeds.shape[-1]
            )

        if self.config.pos_embed:
            if edge_mask is not None and endpoints is not None and endpoints_mask is not None:
                assert (
                    edge_mask.sum() == endpoints_mask.sum()
                ), "edge mask has shape {} but endpoints mask has shape {}".format(
                    edge_mask.sum(), endpoints_mask.sum()
                )
                for i in range(inputs_embeds.shape[0]):
                    _endpoints = endpoints[i][endpoints_mask[i]]
                    if self.config.discretize:
                        _endpoints = _discretize(
                            _endpoints,
                            self.config.bin_num,
                            self.config.verts_bounds,
                            2,
                        )
                    edge_embeds = self.vertex_proj(self.vertex_encoding(_endpoints))
                    inputs_embeds[i, edge_mask[i]] = inputs_embeds[i, edge_mask[i]] + edge_embeds
            if (
                transf_mask is not None
                and transformations is not None
                and transformations_mask is not None
            ):
                assert transf_mask.sum() == transformations_mask.sum()
                for i in range(inputs_embeds.shape[0]):
                    _transformations = transformations[i][transformations_mask[i]]
                    if self.config.discretize:
                        _transformations = _discretize(
                            _transformations, self.config.bin_num, self.config.transf_bounds, 7
                        )
                    transf_embeds = self.transf_proj(
                        torch.cat(
                            [self.trasl_encoding(_transformations[:, :3]), _transformations[:, 3:]],
                            dim=1,
                        )
                    )
                    inputs_embeds[i, transf_mask[i]] = (
                        inputs_embeds[i, transf_mask[i]] + transf_embeds
                    )

        return inputs_embeds

    def forward(
        self,
        *,
        pixel_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        edge_mask: Optional[torch.BoolTensor] = None,
        transf_mask: Optional[torch.BoolTensor] = None,
        endpoints: Optional[torch.FloatTensor] = None,
        endpoints_mask: Optional[torch.BoolTensor] = None,
        transformations: Optional[torch.FloatTensor] = None,
        transformations_mask: Optional[torch.BoolTensor] = None,
    ):
        """Assemble embeddings and run the language model, returning its outputs."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds = self.prepare_forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            edge_mask=edge_mask,
            transf_mask=transf_mask,
            endpoints=endpoints,
            endpoints_mask=endpoints_mask,
            transformations=transformations,
            transformations_mask=transformations_mask,
        )

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    def freeze_vision_tower(self):
        """Freeze the vision tower and image-projection parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.mlp1.parameters():
            param.requires_grad = False

    # Required overrided methods for PreTreainedModel
    @property
    def lm_head(self):
        """The language model's output-embedding (LM) head."""
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        """Return the language model's input embeddings."""
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        """Return the language model's output embeddings."""
        return self.language_model.get_output_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        """Set the language model's input embeddings."""
        return self.language_model.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize the language model's token embeddings to ``new_num_tokens``."""
        self.language_model.resize_token_embeddings(new_num_tokens)


class SwiftTailorForCausalLM(PreTrainedModel, GenerationMixin):
    """Generation wrapper adding edge-parameter decoding on top of the backbone."""

    config_class = SwiftTailorConfig
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: SwiftTailorConfig,
        panel_edge_indices: PanelEdgeTypeIndices,
        gt_stats: StandardizeConfig,
        llm_pretrained_path: Optional[str] = None,
        **kwargs,
    ):
        """Set up conversation template, normalization stats, and the backbone model."""
        super().__init__(config)
        # InternVL config
        self.template = config.template
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

        # AIpparel config
        self.config = config
        self.panel_edge_indices: PanelEdgeTypeIndices = panel_edge_indices
        self.gt_stats = gt_stats
        self.zero_tensor = (
            -torch.tensor(self.gt_stats.vertices.shift) / torch.tensor(self.gt_stats.vertices.scale)
            if self.gt_stats is not None
            else torch.zeros(2)
        )
        self.denormalize_for_loss = config.denormalize_for_loss
        self.edge_loss_weight = config.edge_loss_weight

        self._init_llm_model(llm_pretrained_path, kwargs)

    def _init_llm_model(self, llm_pretrained_path, kwargs):
        """Instantiate the backbone model, from a checkpoint when a path is given."""
        if llm_pretrained_path is None:
            self.model = SwiftTailorModel(
                self.config, panel_edge_indices=self.panel_edge_indices, **kwargs
            )
        else:
            self.model = SwiftTailorModel.from_pretrained(
                llm_pretrained_path,
                config=self.config,
                panel_edge_indices=self.panel_edge_indices,
                **kwargs,
            )

    @property
    def num_image_token(self):
        """Number of visual tokens produced per image."""
        return self.model.num_image_token

    def freeze_vision_tower(self):
        """Freeze the backbone's vision tower."""
        self.model.freeze_vision_tower()

    def prepare_edge_transf_masks_for_inputs(
        self,
        input_ids,
        return_separate=False,
    ):
        """Build boolean masks marking edge and transformation (MOVE) token positions.

        Returns:
            The edge and transform masks, plus a per-index edge mask dict when
            ``return_separate`` is set.
        """
        edge_mask = torch.isin(
            input_ids, torch.tensor(self.panel_edge_indices.get_all_edge_indices()).to(input_ids)
        )
        transf_mask = input_ids == self.panel_edge_indices.get_token_indices(PanelEdgeType.MOVE)

        if return_separate:
            edge_mask_dict = {}
            for ind in self.panel_edge_indices.get_all_edge_indices():
                mask = input_ids == ind
                if not mask.any():
                    continue
                edge_mask_dict[ind] = mask
            return edge_mask, transf_mask, edge_mask_dict
        return edge_mask, transf_mask

    def prepare_edge_panel_mask_dict_for_labels(self, input_ids, labels):
        """Map each panel-edge index to a next-token label mask (shifted by one)."""
        edge_panel_mask_dict = {}
        for index in self.panel_edge_indices.get_all_indices():
            token_mask = labels[:, 1:] == index
            token_mask = torch.cat(
                [
                    token_mask,
                    torch.zeros(
                        (token_mask.shape[0], 1), dtype=torch.bool, device=token_mask.device
                    ),
                ],
                dim=1,
            )
            edge_panel_mask_dict[index] = token_mask
        return edge_panel_mask_dict

    def forward(self, is_training=True, **kwargs):
        """Dispatch to the training or inference forward pass."""
        if is_training:
            return self.training_forward(**kwargs)
        else:
            return self.inference_forward(**kwargs)

    def training_forward(
        self,
        *,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        # AIpparel params
        param_targets: Dict[int, torch.FloatTensor],
        param_target_endpoints: torch.FloatTensor,
        param_target_endpoints_mask: torch.BoolTensor,
        param_target_transformations: torch.FloatTensor,
        param_target_transformations_mask: torch.BoolTensor,
        param_target_masks: Dict[int, torch.BoolTensor],
        **kwargs,
    ):
        """Run the backbone and compute combined cross-entropy and edge-regression loss.

        Returns:
            A dict of losses, predicted parameters, logits, and per-edge-type losses.
        """
        # convert type
        pixel_values = pixel_values.to(dtype=self.dtype)

        edge_panel_mask_dict = self.prepare_edge_panel_mask_dict_for_labels(input_ids, labels)

        if self.config.pos_embed:
            edge_mask, transf_mask = self.prepare_edge_transf_masks_for_inputs(input_ids)
        else:
            edge_mask = None
            transf_mask = None

        output = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=None,
            output_hidden_states=True,  # used for other loss
            # AIpparel params
            edge_mask=edge_mask,
            transf_mask=transf_mask,
            endpoints=param_target_endpoints,
            endpoints_mask=param_target_endpoints_mask,
            transformations=param_target_transformations,
            transformations_mask=param_target_transformations_mask,
        )

        last_hidden_state = output.hidden_states[-1]
        param_preds = {k: torch.zeros_like(v) for k, v in param_targets.items()}
        edge_type_losses = {}
        total_loss = None

        if param_target_masks is not None:
            total_edge_loss = 0
            for ind in param_target_masks.keys():
                panel_edge_index = self.panel_edge_indices.get_index_token(ind).value
                mask = edge_panel_mask_dict[ind]
                if not mask.any():
                    edge_type_losses[f"{panel_edge_index}_loss"] = torch.zeros(1).to(
                        last_hidden_state.device
                    )
                    continue
                panel_embeds = last_hidden_state[mask]
                edge_type = self.panel_edge_indices.get_index_token(ind)
                if edge_type == PanelEdgeType.MOVE:
                    panel_params = self.model.transformation_fc(panel_embeds)
                else:
                    panel_params = self.model.line_curve_fc(panel_embeds)
                    if edge_type == PanelEdgeType.CUBIC:
                        panel_params = panel_params[:, :-2]
                    elif edge_type == PanelEdgeType.ARC:
                        panel_params = torch.cat([panel_params[:, :2], panel_params[:, 6:]], dim=-1)
                    elif edge_type == PanelEdgeType.LINE:
                        panel_params = panel_params[:, :2]
                    elif edge_type == PanelEdgeType.CURVE:
                        panel_params = panel_params[:, :4]
                    elif edge_type == PanelEdgeType.CLOSURE_CURVE:
                        panel_params = panel_params[:, 2:4]
                    elif edge_type == PanelEdgeType.CLOSURE_ARC:
                        panel_params = panel_params[:, 6:]
                    elif edge_type == PanelEdgeType.CLOSURE_CUBIC:
                        panel_params = panel_params[:, 2:6]

                param_preds[ind][param_target_masks[ind]] = panel_params
                if self.gt_stats is not None and self.denormalize_for_loss:
                    param_preds[ind] = denormalize(
                        self.gt_stats, param_preds[ind], is_transf=edge_type == PanelEdgeType.MOVE
                    )
                    param_targets[ind] = denormalize(
                        self.gt_stats, param_targets[ind], is_transf=edge_type == PanelEdgeType.MOVE
                    )
                loss = torch.sum((param_preds[ind] - param_targets[ind]) ** 2, -1).sum(1) / (
                    torch.sum(param_target_masks[ind], 1) + 1e-5
                )
                total_edge_loss += loss

                edge_type_losses[f"{panel_edge_index}_loss"] = loss.mean()

            total_loss = self.edge_loss_weight * total_edge_loss
        logits = output.logits
        ce_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            vocab_size = self.model.language_model.config.vocab_size
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss = loss_fct(shift_logits, shift_labels)
        if ce_loss is not None:
            total_loss += ce_loss

        return_dict = {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "edge_loss": total_edge_loss,
            "params": param_preds,
            "logits": logits,
        }
        return_dict.update(edge_type_losses)
        return return_dict

    # Required overrided methods for PreTrainedModel
    @property
    def lm_head(self):
        """The backbone's output-embedding (LM) head."""
        return self.model.get_output_embeddings()

    def get_input_embeddings(self):
        """Return the backbone's input embeddings."""
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        """Return the backbone's output embeddings."""
        return self.model.get_output_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        """Set the backbone's input embeddings."""
        return self.model.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize the token embeddings matrix to ``new_num_tokens``."""
        self.model.resize_token_embeddings(new_num_tokens)

    # Inference implementation
    @torch.no_grad()
    def inference_forward(self, **kwargs):
        """Run a single backbone forward pass without gradients."""
        return self.model(
            **kwargs,
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        """Carry the last hidden state forward across generation steps."""
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        model_kwargs["last_hidden_state"] = outputs.hidden_states[-1]
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        last_hidden_state=None,
        edge_mask=None,
        transf_mask=None,
        endpoints=None,
        endpoints_mask=None,
        transformations=None,
        transformations_mask=None,
        is_training=False,  # This flag leads to generation mode
        **kwargs,
    ):
        """Assemble per-step generation inputs, decoding edge/transform embeddings.

        Returns:
            The model-input dict for the next generation step.
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]
            last_hidden_state = last_hidden_state[:, -1:, :]
        edge_mask, transf_mask, edge_mask_dict = self.prepare_edge_transf_masks_for_inputs(
            input_ids, return_separate=True
        )
        if self.config.pos_embed and last_hidden_state is not None:
            endpoints, endpoints_mask = None, None
            transformations, transformations_mask = None, None
            if edge_mask.any():
                assert edge_mask.shape[1] == last_hidden_state.shape[1]
                endpoints = torch.zeros(
                    last_hidden_state.shape[0], last_hidden_state.shape[1], 2
                ).to(last_hidden_state)
                endpoints_mask = (
                    torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1])
                    .to(last_hidden_state)
                    .bool()
                )
                for ind, mask in edge_mask_dict.items():
                    if not mask.any():
                        continue
                    endpoints_mask |= mask
                    edge_type = self.panel_edge_indices.get_index_token(ind)
                    edge_embeds = last_hidden_state[mask]
                    if edge_type.is_closure():
                        endpoints[mask] = self.zero_tensor.to(edge_embeds)
                    if edge_type == PanelEdgeType.CUBIC:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                    elif edge_type == PanelEdgeType.ARC:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                    elif edge_type == PanelEdgeType.LINE:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                    elif edge_type == PanelEdgeType.CURVE:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]

            if transf_mask.any():
                assert transf_mask.shape[1] == last_hidden_state.shape[1]
                transformations = torch.zeros(
                    last_hidden_state.shape[0], last_hidden_state.shape[1], 7
                ).to(last_hidden_state)
                transformations_mask = (
                    torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1])
                    .to(last_hidden_state)
                    .bool()
                )
                transf_embeds = last_hidden_state[transf_mask]
                transformations[transf_mask] = self.model.transformation_fc(transf_embeds)
                transformations_mask[transf_mask] = True

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "edge_mask": edge_mask,
                "transf_mask": transf_mask,
                "endpoints": endpoints,
                "endpoints_mask": endpoints_mask,
                "transformations": transformations,
                "transformations_mask": transformations_mask,
                "is_training": is_training,
            }
        )
        return model_inputs

    @torch.no_grad()
    def chat(
        self,
        *,
        pixel_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        edge_mask: Optional[torch.BoolTensor] = None,
        transf_mask: Optional[torch.BoolTensor] = None,
        endpoints: Optional[torch.FloatTensor] = None,
        endpoints_mask: Optional[torch.BoolTensor] = None,
        transformations: Optional[torch.FloatTensor] = None,
        transformations_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        """Generate a pattern token sequence and decode its edge parameters.

        Returns:
            A dict with the generated ``output_ids`` and per-edge predicted ``params``.
        """
        outputs = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            edge_mask=edge_mask,
            transf_mask=transf_mask,
            endpoints=endpoints,
            endpoints_mask=endpoints_mask,
            transformations=transformations,
            transformations_mask=transformations_mask,
            output_hidden_states=True,  # used to prepare generation
            return_dict_in_generate=True,
            **kwargs,
        )

        # Extract the output ids
        output_ids = outputs.sequences

        edge_panel_mask_dict = {}
        for ind in self.panel_edge_indices.get_all_indices():
            if self.panel_edge_indices.get_index_param_num(ind) == 0:
                continue
            token_mask = output_ids[:, 1:] == ind
            token_mask[:, : input_ids.shape[1] - 1] = False
            edge_panel_mask_dict[ind] = token_mask

        last_hidden_state = torch.cat(
            [outputs.hidden_states[i][-1] for i in range(len(outputs.hidden_states))], dim=1
        )

        param_preds = {}
        for ind, mask in edge_panel_mask_dict.items():
            if mask.sum() == 0:
                continue
            panel_embeds = last_hidden_state[mask]
            edge_type = self.panel_edge_indices.get_index_token(ind)
            if edge_type == PanelEdgeType.MOVE:
                panel_params = self.model.transformation_fc(panel_embeds)
            else:
                panel_params = self.model.line_curve_fc(panel_embeds)
                if edge_type == PanelEdgeType.CUBIC:
                    panel_params = panel_params[:, :-2]
                elif edge_type == PanelEdgeType.ARC:
                    panel_params = torch.cat([panel_params[:, :2], panel_params[:, 6:]], dim=-1)
                elif edge_type == PanelEdgeType.LINE:
                    panel_params = panel_params[:, :2]
                elif edge_type == PanelEdgeType.CURVE:
                    panel_params = panel_params[:, :4]
                elif edge_type == PanelEdgeType.CLOSURE_CURVE:
                    panel_params = panel_params[:, 2:4]
                elif edge_type == PanelEdgeType.CLOSURE_ARC:
                    panel_params = panel_params[:, 6:]
                elif edge_type == PanelEdgeType.CLOSURE_CUBIC:
                    panel_params = panel_params[:, 2:6]

            param_preds[ind] = panel_params

        return {"output_ids": output_ids, "params": param_preds}
