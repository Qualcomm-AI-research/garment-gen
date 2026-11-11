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
# Based on: https://github.com/georgeNakayama/AIpparel-Code/blob/master/data/datasets/utils.py

from functools import partial

import numpy as np
import torch

from .constants import IGNORE_TOKEN_ID, IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN


def discretize(
    params: np.ndarray, bin_size: int, shift: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    params: np.ndarray = (params - shift) / scale
    params = np.clip(params, 0, 1) * bin_size
    params = params.astype(int).clip(0, bin_size - 1)
    return params


DEFAULT_IMAGE_PLACEHOLDER_TOKEN = "<image>"
DEFAULT_IMAGE_CONTEXT_TOKEN = IMG_CONTEXT_TOKEN
DEFAULT_IM_START_TOKEN = IMG_START_TOKEN
DEFAULT_IM_END_TOKEN = IMG_END_TOKEN
DEFAULT_PLACEHOLDER_TOKEN = "<placeholder>"

HMR_SHORT_QUESTION_LIST = [
    "I have a description of a person's pose, can you give the SMPL pose of this person?",
    "Give you a word descrption of a human, please output the SMPL pose.",
    "Describe what this perosn is doing using SMPL pose.",
    "What's the SMPL pose of this person?",
    "Use SMPL pose to describe this person's behavior.",
]

DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST = [
    "I have a word description of a piece of garment, can you provide the sewing pattern? {sent}",
    "There is a garment like this: {sent} Please output this sewing pattern.",
    "{sent} Give the sewing pattern.",
    "What's the sewing pattern? {sent}",
    "Describe the garment as a sewing pattern. {sent}",
    "Sewing pattern is described as words: {sent} The sewing pattern is?",
    "I have a garment in this style: {sent} Can you provide the sewing pattern?",
    "Can you provide a sewing pattern for this garment? {sent}",
    "What is the sewing pattern of this garment? {sent}",
    "Sewing pattern can be described as words: {sent} And it can also be described as a sewing pattern as tokens. Can you output this?",
]

SPECULATIVE_TEXT_SHORT_QUESTION_LIST = [
    "Please output a garment that can be worn in this situation as a sewing pattern. {sent}",
    "Can you provide a garment that can be worn in this situation as a sewing pattern? {sent}",
    "What is an example of a piece of garment that can be worn in this situation? {sent} Output as a sewing pattern.",
    "I want to dress up for the following occasion. {sent} Can you provide a sewing pattern that can be worn?",
    "What would be a garment that can be worn in this situation? Output in the form of a sewing pattern. {sent}",
    "I want a piece of garment most suitable for this occasion. {sent} Can you provide a sewing pattern that can be worn?",
]

SHORT_QUESTION_WITH_TEXT_LIST = [
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN
    + "\n"
    + "The person in the image is wearing a garment with details given in words: {sent} Can you provide the sewing pattern of the garment?",
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN
    + "\n"
    + "The garment in the image is described as: {sent} Output the sewing pattern of the garment.",
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN
    + "\n"
    + "The person is wearing in the this style: {sent} Please respond with sewing pattern.",
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN
    + "\n"
    + "The garment in the image is in this style: {sent} Please output sewing pattern.",
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN
    + "\n"
    + "There is a person in the middle of the image with details given in words: {sent} Use a sewing pattern to describe the person's garment.",
]

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN
    + "\n"
    + "Can you predict the sewing pattern of the person in this image?",
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN
    + "\n"
    + "There is a person in the middle of the image, please output this person's sewing pattern.",
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN
    + "\n"
    + "What is the garment in this image? Please respond with sewing pattern.",
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN
    + "\n"
    + "What is the person wearing in this image? Please output sewing pattern.",
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN
    + "\n"
    + "There is a person in the middle of the image, use sewing pattern to describe the person's garment.",
]

EDITING_QUESTION_LIST = [
    "I have a sewing pattern: {pattern} Edit the sewing pattern according to the instruction. {sent}",
    "I want to edit the sewing pattern {pattern} based on the following instruction: {sent} Output in the form of a sewing pattern.",
    "Modify the sewing pattern {pattern} based on the following guidance: {sent} Respond with a sewing pattern.",
    "I have a sewing pattern: {pattern} Please provide a new sewing pattern based on the instruction: {sent}",
    "How would the sewing pattern {pattern} look like if I edit it based on the following instruction? {sent} Respond with a new sewing pattern.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output sewing pattern and explain its style.",
    "Please output sewing pattern and explain when to wear it.",
    "Please output sewing pattern and give some explanation.",
]

ANSWER_LIST = [
    "It is {pattern}.",
    "It is a sewing pattern for {pattern}.",
    "Sure, {pattern}.",
    "Sure, it is {pattern}.",
    "Sure, the sewing pattern is {pattern}.",
    "{pattern}.",
    "The sewing pattern is {pattern}.",
    "The sewing pattern of the person is {pattern}.",
    "The sewing pattern of this person's garment is {pattern}.",
]


# ------------------ Transforms ----------------
def _dict_to_tensors(dict_obj):  # helper
    """convert a dictionary with numeric values into a new dictionary with torch tensors"""
    new_dict = dict.fromkeys(dict_obj.keys())
    for key, value in dict_obj.items():
        if value is None:
            new_dict[key] = torch.Tensor()
        elif isinstance(value, dict):
            new_dict[key] = _dict_to_tensors(value)
        elif isinstance(value, str):  # no changes for strings
            new_dict[key] = value
        elif isinstance(value, np.ndarray):
            new_dict[key] = torch.from_numpy(value)

            # TODO more stable way of converting the types (or detecting ints)
            if value.dtype not in [int, np.int64, bool]:
                # cast all doubles and other stuff to floats
                new_dict[key] = new_dict[key].float()
        else:
            # just try directly, if nothing else works
            new_dict[key] = torch.tensor(value)
    return new_dict


# Custom transforms -- to tensor
class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return _dict_to_tensors(sample)


# --------------------- Utils ---------------------
def factory_fn(fn_path, *args, **kwargs):
    """A factory function to load a function from a file."""
    module_name, fn_name = fn_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[fn_name])
    fn = getattr(module, fn_name)
    fn = partial(fn, *args, **kwargs)
    return fn
