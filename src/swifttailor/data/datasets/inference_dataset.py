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
# Based on: https://github.com/georgeNakayama/AIpparel-Code/blob/master/data/datasets/inference_dataset.py

import json
import random
from collections import defaultdict
from typing import Dict, List, Optional

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

# My
from swifttailor.data.garment_tokenizers.gcd_garment_tokenizer import GCDGarmentTokenizer
from swifttailor.data.patterns.panel_classes import PanelClasses
from swifttailor.data.patterns.pattern_converter import NNSewingPattern
from swifttailor.patternmaker.internvl.conversation import conv_templates as conversation_lib


class InferenceDataset(Dataset):
    def __init__(
        self,
        *,
        inference_json: str,
        garment_tokenizer: GCDGarmentTokenizer,
        panel_classification: Optional[str] = NotImplemented,
        # Image processing params
        image_size: int = 448,
        max_num_patches: int = 12,
        conv_template: str = "internvl2_5",
    ):

        self.panel_classifier = None
        self.panel_classification = panel_classification

        #################################
        # init from the basedataset class

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

        datapoints_names = json.load(open(inference_json, "r"))
        for datapoint_dict in datapoints_names:
            self.datapoints_names.append((datapoint_dict["type"], datapoint_dict["inputs"]))

        self.panel_classifier = PanelClasses(classes_file=panel_classification)
        self.panel_classes = self.panel_classifier.classes

        print("The panel classes in this dataset are :", self.panel_classes)

    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)

    def _fetch_pattern(self, spec_path):
        """Fetch the pattern and image for the given index"""
        gt_pattern = NNSewingPattern(spec_path)
        garment_name = spec_path.split("/")[-1].split(".")[0]
        gt_pattern.name = garment_name
        return gt_pattern

    def _parepare_image(self, image_path):
        """Fetch the image for the given index"""
        image = load_image(image_path, self.image_size, self.max_num_patches)
        return image, image_path

    def get_mode_names(self):
        return ["image", "description", "occasion", "text_image", "editing"]

    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data.
        Does not support list indexing"""

        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        data_type, data_inputs = self.datapoints_names[idx]
        assert data_type in ["image", "description", "occasion", "text_image", "editing"]
        description = None
        editing_spec_file = None
        editing_caption = None
        image_path = ""
        image_clip = None
        if data_type in ["image", "text_image"]:
            image_path = data_inputs["image_path"]
            image_clip, image_path = self._parepare_image(image_path)
        if data_type == "editing":
            editing_spec_file = data_inputs["spec_path"]
            description = data_inputs["description"]
        if data_type in ["description", "occasion"]:
            description = data_inputs["description"]
        if data_type == "editing":
            editing_spec_file = data_inputs["spec_path"]
            edited_pattern = NNSewingPattern(
                editing_spec_file, panel_classifier=self.panel_classifier, template_name=None
            )
            edited_pattern_dict = self.garment_tokenizer.encode(edited_pattern)
            editing_caption = data_inputs["description"]

        out_pattern_dict = {}
        out_pattern = []
        if data_type == "image":
            # image_only
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
            question_pattern_dict = {}
        elif data_type == "description":
            # descriptive text_only
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.descriptive_text_question_list).format(
                    sent=description
                )
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append(answer_template)
            question_pattern_dict = {}
        elif data_type == "occasion":
            # speculative text_only
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.speculative_text_question_list).format(
                    sent=description
                )
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append(answer_template)
            question_pattern_dict = {}
        elif data_type == "text_image":
            # image_text
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.text_image_question_list).format(
                    sent=description
                )
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append(answer_template)
            question_pattern_dict = {}
        elif data_type == "editing":
            # garment_editing
            before_pattern_dict = edited_pattern_dict
            edited_pattern.name = "before_" + edited_pattern.name
            out_pattern = [edited_pattern]
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.editing_question_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN, sent=editing_caption
                )
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append(answer_template)
            out_pattern_dict = {k: before_pattern_dict[k] for k in before_pattern_dict.keys()}
            question_pattern_dict = before_pattern_dict
        else:
            raise ValueError(f"Invalid sample type: {data_type}")

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
            image_clip,
            conversations,
            question_only_convs,
            questions,
            out_pattern,
            {"image": 0, "description": 1, "occasion": 2, "text_image": 3, "editing": 4}[data_type],
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
