# Copyright (c) 2024 Maria Korosteleva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes from Qualcomm Technologies, Inc. are provided under the following license:
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Based on: https://github.com/georgeNakayama/AIpparel-Code/blob/master/data/datasets/gcd_mm_dataset.py

import hashlib
import json
import os
import random
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from swifttailor.data.datasets.preprocessing_utils import load_image
from swifttailor.data.datasets.utils import (
    ANSWER_LIST,
    DEFAULT_PLACEHOLDER_TOKEN,
    DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST,
    EDITING_QUESTION_LIST,
    SHORT_QUESTION_LIST,
    SHORT_QUESTION_WITH_TEXT_LIST,
    SPECULATIVE_TEXT_SHORT_QUESTION_LIST,
)

from swifttailor.data.garment_tokenizers.gcd_garment_tokenizer import GCDGarmentTokenizer
from swifttailor.data.patterns.panel_classes import PanelClasses
from swifttailor.data.patterns.pattern_converter import NNSewingPattern
from swifttailor.patternmaker.internvl.conversation import conv_templates as conversation_lib


class DataType(Enum):
    IMAGE = 0
    DESCRIPTIVE_TEXT = 1
    SPECULATIVE_TEXT = 2
    IMAGE_TEXT = 3
    EDITING = 4

    def requires_text(self):
        return self in [DataType.IMAGE_TEXT, DataType.DESCRIPTIVE_TEXT, DataType.SPECULATIVE_TEXT]

    @classmethod
    def list(cls):
        return list(map(lambda c: c, cls))


def set_seed_from_name(name: str):
    # Hash the name to a reproducible integer
    hash_bytes = hashlib.sha256(name.encode("utf-8")).digest()
    # Convert first 4 bytes to an integer (fits into 32-bit)
    seed = int.from_bytes(hash_bytes[:4], "big")

    # Set both numpy and random seed
    np.random.seed(seed)
    random.seed(seed)


class GCDMM(Dataset):
    def __init__(
        self,
        *,
        root_dir: str,
        editing_dir: str,
        caption_dir: str,
        sampling_rate: List[int],
        editing_flip_prob: float,
        load_by_dataname: List[Tuple[str, str, str]],
        garment_tokenizer: GCDGarmentTokenizer,
        panel_classification: Optional[str] = NotImplemented,
        # Image processing params
        image_size: int = 448,
        max_num_patches: int = 12,
        conv_template: str = "internvl2_5",
        is_training: bool = True,
    ):
        self.is_training = is_training
        self.editing_dir = editing_dir
        self.editing_flip_prob = editing_flip_prob
        self.caption_dir = caption_dir
        self.sampling_rate = sampling_rate
        self.panel_classifier = None
        self.panel_classification = panel_classification

        #################################
        # init from the basedataset class
        self.root_path = Path(root_dir)

        self.datapoints_names = []
        self.panel_classes = []
        self.dict_panel_classes = defaultdict(int)
        self.garment_tokenizer = garment_tokenizer

        # Image processing params
        self.image_size = image_size
        self.max_num_patches = max_num_patches

        self.short_question_list = SHORT_QUESTION_LIST
        self.descriptive_text_question_list = DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST
        self.speculative_text_question_list = SPECULATIVE_TEXT_SHORT_QUESTION_LIST
        self.text_image_question_list = SHORT_QUESTION_WITH_TEXT_LIST
        self.editing_question_list = EDITING_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.conv_template = conv_template

        # collect information about data folders:

        if isinstance(load_by_dataname, str) and load_by_dataname.endswith(".txt"):
            with open(load_by_dataname, "r") as f:
                load_by_dataname = [line.strip() for line in f.readlines()]
        # the full GarmentCodeData release uses (default_body/data/rand_xxx).
        # Keep names as-is: default_body/rand_xxx → root_dir/default_body/rand_xxx
        self.datapoints_names = list(load_by_dataname)

        self.panel_classifier = PanelClasses(classes_file=panel_classification)
        self.panel_classes = self.panel_classifier.classes

        print("The panel classes in this dataset are :", self.panel_classes)

        self.gt_cached = {}
        self.gt_caching = True

        ########################################
        # end of init from the basedataset class

        self.gt_cached = {}

    def set_training(self, is_training: bool):
        self.is_training = is_training

    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)

    def _fetch_pattern(self, idx):
        """Fetch the pattern and image for the given index"""
        _, garment_names, garment_spec = self.datapoints_names[idx]
        gt_pattern = NNSewingPattern(garment_spec)
        gt_pattern.name = "_".join(garment_names)
        return gt_pattern

    def _parepare_image(self, image_paths):
        """Fetch the image for the given index
        Args:
            image_paths (List[str]):
                List of rendered image paths for the garment.
        Returns:
            images:
                Processed patches of the chosen image and its resized version.
            image_path: Path to the chosen image.
        """
        # Each sample contains multiple views (front, back,...)
        # Pick one randomly
        if self.is_training:
            image_path = random.choice(image_paths)
        else:  # for evaluation, use front image
            if "_render_front" in image_paths[0]:
                image_path = image_paths[0]
            else:
                image_path = image_paths[-1]
        images = load_image(image_path, self.image_size, self.max_num_patches)
        return images, image_path

    def get_mode_names(self):
        return ["image", "description", "occasion", "text_image", "editing"]

    def __getitem__(self, idx):
        """Get item by index"""
        max_try = 10
        for _ in range(max_try):
            try:
                # Try to get the item
                return self._my_getitem(idx)
            except Exception as e:
                print(f"Error while getting item {idx}: {e}")
                if str(e) == "string indices must be integers":
                    import traceback

                    traceback.print_exc()
            finally:
                # If error, try to get the next item
                idx = (idx + 1) % self.__len__()

    def _my_getitem(self, idx):
        """Called when indexing: read the corresponding data.
        Does not support list indexing"""
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        datapoint_name = self.datapoints_names[idx]
        data_name = datapoint_name.split("/")[-1]

        # deterministic sampling
        set_seed_from_name(data_name)

        image_paths = [
            os.path.join(self.root_path, datapoint_name, f"{data_name}_render_back.png"),
            os.path.join(self.root_path, datapoint_name, f"{data_name}_render_front.png"),
        ]
        if data_name in self.gt_cached:
            (
                gt_pattern,
                pattern_dict,
                edited_pattern,
                edited_pattern_dict,
                editing_captions,
                captions,
            ) = self.gt_cached[data_name]
        else:
            spec_file = os.path.join(
                self.root_path, datapoint_name, f"{data_name}_specification_shifted.json"
            )
            gt_pattern = NNSewingPattern(
                spec_file, panel_classifier=self.panel_classifier, template_name=data_name
            )
            gt_pattern.name = data_name
            pattern_dict = self.garment_tokenizer.encode(gt_pattern)

            editing_spec_file = os.path.join(
                self.editing_dir, data_name, "edited_specification.json"
            )
            editing_caption_json = os.path.join(self.editing_dir, data_name, "editing_caption.json")
            if (not os.path.exists(editing_spec_file)) or (
                not os.path.exists(editing_caption_json)
            ):
                edited_pattern = None
                edited_pattern_dict = None
                editing_captions = None
            else:
                edited_pattern = NNSewingPattern(
                    editing_spec_file,
                    panel_classifier=self.panel_classifier,
                    template_name=data_name,
                )

                edited_pattern.name = data_name
                edited_pattern_dict = self.garment_tokenizer.encode(edited_pattern)
                editing_captions = json.load(open(editing_caption_json, "r"))

            caption_json = os.path.join(self.caption_dir, datapoint_name, "captions.json")
            if os.path.exists(caption_json):
                captions = json.load(open(caption_json, "r"))
            else:
                captions = None

            self.gt_cached[data_name] = (
                gt_pattern,
                pattern_dict,
                edited_pattern,
                edited_pattern_dict,
                editing_captions,
                captions,
            )

        images = None
        image_path = ""
        sample_type = np.random.choice(list(DataType), p=self.sampling_rate)
        if sample_type == DataType.EDITING and edited_pattern is None:
            # no editing if there is no edited pattern
            sample_type = np.random.choice(list(DataType)[:-1])
            assert sample_type != DataType.EDITING  # safety check
        if DataType.requires_text(sample_type) and captions is None:
            # no text if there is no caption
            sample_type = DataType.IMAGE
        if sample_type == DataType.IMAGE:
            # image_only
            images, image_path = self._parepare_image(image_paths)
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.short_question_list)
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append(answer_template)
            out_pattern_dict = pattern_dict
            question_pattern_dict = {}
            out_pattern = [gt_pattern]
        elif sample_type == DataType.DESCRIPTIVE_TEXT:
            # descriptive text_only
            descriptive_text = captions["description"]
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.descriptive_text_question_list).format(
                    sent=descriptive_text
                )
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append(answer_template)
            out_pattern_dict = pattern_dict
            question_pattern_dict = {}
            out_pattern = [gt_pattern]
        elif sample_type == DataType.SPECULATIVE_TEXT:
            # speculative text_only
            speculative_text = captions["occasion"]
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.speculative_text_question_list).format(
                    sent=speculative_text
                )
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append(answer_template)
            out_pattern_dict = pattern_dict
            question_pattern_dict = {}
            out_pattern = [gt_pattern]
        elif sample_type == DataType.IMAGE_TEXT:
            # image_text
            descriptive_text = captions["description"]
            images, image_path = self._parepare_image(image_paths)
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.text_image_question_list).format(
                    sent=descriptive_text
                )
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append(answer_template)
            out_pattern_dict = pattern_dict
            question_pattern_dict = {}
            out_pattern = [gt_pattern]
        elif sample_type == DataType.EDITING:
            # garment_editing
            if random.random() > self.editing_flip_prob:
                before_pattern_dict = pattern_dict
                after_pattern_dict = edited_pattern_dict
                editing_text = editing_captions["editing_description_forward"]
                gt_pattern.name = "before_" + gt_pattern.name
                edited_pattern.name = "after_" + edited_pattern.name
                out_pattern = [gt_pattern, edited_pattern]
            else:
                before_pattern_dict = edited_pattern_dict
                after_pattern_dict = pattern_dict
                editing_text = editing_captions["editing_description_reverse"]
                edited_pattern.name = "before_" + edited_pattern.name
                gt_pattern.name = "after_" + gt_pattern.name
                out_pattern = [edited_pattern, gt_pattern]
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.editing_question_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN, sent=editing_text
                )
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append(answer_template)
            out_pattern_dict = {
                k: before_pattern_dict[k] + after_pattern_dict[k]
                for k in before_pattern_dict.keys()
            }
            question_pattern_dict = before_pattern_dict
        else:
            raise ValueError(f"Invalid sample type: {sample_type}")

        conversations = []
        question_only_convs = []
        conv = conversation_lib[self.conv_template]

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], "")
            question_only_convs.append(conv.get_prompt())
            i += 1

        return (
            # dict with keys
            # ['description', 'params', 'endpoints', 'transformations']
            out_pattern_dict,
            # similar but applied for questions (empty if not editing)
            question_pattern_dict,
            image_path,
            images,
            conversations,  # full with llm special tokens
            question_only_convs,  # question with llm special tokens
            questions,  # question without llm special tokens
            out_pattern,  # NNSewingPattern object
            sample_type.value,  # input type (IMAGE_TEXT,...)
        )

    @property
    def panel_edge_type_indices(self):
        return self.garment_tokenizer.panel_edge_type_indices

    @property
    def gt_stats(self):
        return self.garment_tokenizer.gt_stats

    def get_all_token_names(self):
        return self.garment_tokenizer.get_all_token_names()

    def set_token_indices(self, token2idx: Dict[str, int]):
        return self.garment_tokenizer.set_token_indices(token2idx)

    def evaluate_patterns(
        self, pred_patterns: List[NNSewingPattern], gt_patterns: List[NNSewingPattern]
    ):
        return self.garment_tokenizer.evaluate_patterns(pred_patterns, gt_patterns)

    def get_item_infos(self, index):
        return self.datapoints_names[index]

    def decode(self, output_ids: torch.Tensor, tokenizer: PreTrainedTokenizer):
        """Decode output ids to text"""
        return self.garment_tokenizer.decode(output_ids, tokenizer)

    def split_from_dict(self, split_dict):
        """
        Reproduce the data split in the provided dictionary:
        the elements of the currect dataset should
        play the same role as in provided dict
        """
        train_ids, valid_ids, test_ids = [], [], []

        training_datanames = split_dict["train"]
        valid_datanames = split_dict["validation"]
        test_datanames = split_dict["test"]

        for idx in range(len(self.datapoints_names)):
            # usually the largest, so check first
            if self.datapoints_names[idx] in training_datanames:
                train_ids.append(idx)
            elif self.datapoints_names[idx] in test_datanames:
                test_ids.append(idx)
            elif self.datapoints_names[idx] in valid_datanames:
                valid_ids.append(idx)
            else:
                continue

            if idx % 1000 == 0:
                print(
                    f"progress {idx}, #Train_Ids={len(train_ids)}, "
                    f"#Valid_Ids={len(valid_ids)}, #Test_Ids={len(test_ids)}"
                )

        return (
            [self.datapoints_names[i] for i in train_ids],
            [self.datapoints_names[i] for i in valid_ids],
            [self.datapoints_names[i] for i in test_ids] if len(test_ids) > 0 else None,
        )
