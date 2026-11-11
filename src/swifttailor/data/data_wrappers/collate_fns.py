# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Batch collation helpers turning samples into model-ready tensors."""

from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional

import torch
import transformers

from swifttailor.data.patterns.pattern_converter import NNSewingPattern
from swifttailor.patternmaker.internvl.conversation import conv_templates as conversation_lib

from ..datasets.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_CONTEXT_TOKEN,
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN,
    DEFAULT_PLACEHOLDER_TOKEN,
    IGNORE_TOKEN_ID,
)


def _extract_batch(batch):
    """Unzip raw batch tuples into per-field lists of answer, question, and vision data."""
    sample_type_list: List[int] = []

    # Answer params
    encoded_pattern_list = []
    pattern_param_list = []
    pattern_endpoints_list = []
    pattern_transf_list = []

    # Question params (non-empty when sample type is editting)
    question_pattern_list = []
    question_pattern_param_list = []
    question_pattern_endpoints_list = []
    question_pattern_transf_list = []

    # Vision information
    image_path_list: List[str] = []
    images_list: List[torch.Tensor] = []

    gt_pattern_list: List[NNSewingPattern] = []

    conversation_list: List[str] = []  # full with llm special tokens
    question_conv_list: List[str] = []  # question with llm special tokens
    questions_list: List[str] = []  # question without llm special tokens
    offset_list: List[int] = [0]
    cnt: int = 0

    for (
        pattern_dict,
        question_pattern_dict,
        image_path,
        images,
        conversations,
        question_only_convs,
        questions,
        gt_pattern,
        sample_type,
    ) in batch:
        sample_type_list.append(sample_type)

        # Answer params extraction
        pattern_description = pattern_dict.get("description", [])
        if pattern_description is not None:
            encoded_pattern_list.append(pattern_description)
        else:
            encoded_pattern_list.append([])
        pattern_param = pattern_dict.get("params", None)
        if pattern_param is not None:
            pattern_param_list.append(pattern_param)
        else:
            pattern_param_list.append([{}])
        pattern_endpoint = pattern_dict.get("endpoints", None)
        if pattern_endpoint is not None:
            pattern_endpoint = torch.cat(pattern_endpoint)
        else:
            pattern_endpoint = torch.zeros(0, 2)
        pattern_endpoints_list.append(pattern_endpoint)
        pattern_transf = pattern_dict.get("transformations", None)
        if pattern_transf is not None:
            pattern_transf = torch.cat(pattern_transf)
        else:
            pattern_transf = torch.zeros(0, 7)
        pattern_transf_list.append(pattern_transf)

        # Question params extraction
        question_pattern_list.append(question_pattern_dict.get("description", []))
        question_pattern_param = question_pattern_dict.get("params", None)
        if question_pattern_param is not None:
            question_pattern_param_list.append(question_pattern_param)
        else:
            question_pattern_param_list.append([{}])
        question_pattern_endpoint = question_pattern_dict.get("endpoints", None)
        if question_pattern_endpoint is not None:
            question_pattern_endpoint = torch.cat(question_pattern_endpoint)
        else:
            question_pattern_endpoint = torch.zeros(0, 2)
        question_pattern_endpoints_list.append(question_pattern_endpoint)
        question_pattern_transf = question_pattern_dict.get("transformations", None)
        if question_pattern_transf is not None:
            question_pattern_transf = torch.cat(question_pattern_transf)
        else:
            question_pattern_transf = torch.zeros(0, 7)
        question_pattern_transf_list.append(question_pattern_transf)

        # Vision information
        image_path_list.append(image_path)
        images_list.append(images)

        # Conversations
        conversation_list.extend(conversations)
        question_conv_list.extend(question_only_convs)
        questions_list.append(questions)
        gt_pattern_list.append(gt_pattern)

        cnt += len(conversations)
        offset_list.append(cnt)

    return (
        sample_type_list,
        encoded_pattern_list,
        pattern_param_list,
        pattern_endpoints_list,
        pattern_transf_list,
        question_pattern_list,
        question_pattern_param_list,
        question_pattern_endpoints_list,
        question_pattern_transf_list,
        image_path_list,
        images_list,
        gt_pattern_list,
        conversation_list,
        question_conv_list,
        questions_list,
        offset_list,
    )


def _replace_image_placeholder(
    conversation: str,
    images: torch.Tensor,
    num_image_token: int,
):
    """Expand the single ``<image>`` placeholder into per-image context tokens.

    Args:
        conversation: Conversation text holding the placeholder.
        images: Image tensor shaped ``[num_images, C, H, W]``.
        num_image_token: Number of context tokens emitted per image.

    Returns:
        The conversation with the placeholder replaced.
    """
    num_image = images.shape[0] if images is not None else 0
    replace_token = DEFAULT_IMAGE_CONTEXT_TOKEN * num_image * num_image_token
    if num_image > 0:
        replace_token = (
            DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        )  # <img_start><image_context><image_context>...<img_end>
    else:
        replace_token = ""

    n_img_placeholder_token = conversation.count(DEFAULT_IMAGE_PLACEHOLDER_TOKEN)
    assert n_img_placeholder_token <= 1, (
        f"Image placeholder appears {n_img_placeholder_token} times. "
        f"Expected at most once.\n"
        f"Image placeholder: {DEFAULT_IMAGE_PLACEHOLDER_TOKEN}.\n"
        f"Conversation: {conversation}."
    )
    conversation = conversation.replace(DEFAULT_IMAGE_PLACEHOLDER_TOKEN, replace_token)

    return conversation


def _replace_pattern_placeholder(
    conversation: str,
    patterns: List[str],
):
    """Substitute each pattern placeholder token in order with the given patterns.

    Args:
        conversation: Conversation text holding the placeholders.
        patterns: Patterns to splice in, one per placeholder.

    Returns:
        The conversation with placeholders replaced.
    """
    n_placeholders = conversation.count(DEFAULT_PLACEHOLDER_TOKEN)
    assert n_placeholders == len(patterns), (
        f"Number of placeholders ({n_placeholders}) does not match "
        f"the number of patterns ({len(patterns)}).\n"
        f"Conversation: {conversation}\n"
        f"Patterns: {patterns}"
    )
    for pattern in patterns:
        conversation = conversation.replace(DEFAULT_PLACEHOLDER_TOKEN, "".join(pattern), 1)
    return conversation


def replace_placeholder(
    *,
    conversation_list: List[str],
    patterns_list: List[List[str]],
    images_list: List[torch.Tensor],
    num_image_token: int,
):
    """Replace pattern and image placeholders in every conversation in place."""
    for i in range(len(conversation_list)):
        conversation_list[i] = _replace_pattern_placeholder(
            conversation_list[i],
            patterns_list[i],
        )
        conversation_list[i] = _replace_image_placeholder(
            conversation_list[i],
            images_list[i],
            num_image_token,
        )

    return conversation_list


def tokenize_conversation(
    conversation_list: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> torch.Tensor:
    """Tokenize conversations and right-pad them to a common length.

    Returns:
        Padded ``input_ids`` shaped ``[batch_size, max_sequence_length]``.
    """
    tokenized_conversations: List[torch.Tensor] = [
        torch.tensor(conv_token_ids) for conv_token_ids in tokenizer(conversation_list).input_ids
    ]
    padded_conversations: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        tokenized_conversations,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side="right",
    )
    return padded_conversations


def preprocess_embedding_params(params: List[torch.Tensor]):
    """Pad a list of parameter tensors and build matching validity masks.

    Returns:
        The padded parameters and their boolean masks.
    """
    masks: List[torch.Tensor] = [
        # exclude information dim (last dim)
        torch.ones(param.shape[:-1], dtype=torch.bool)
        for param in params
    ]
    padded_params: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        params, batch_first=True, padding_value=0.0
    )
    padded_masks: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        masks, batch_first=True, padding_value=False
    )

    return padded_params, padded_masks


def prepare_conversation(
    *,
    # Text
    conversation_list: List[str],
    pattern_list: List[List[str]],
    tokenizer: transformers.PreTrainedTokenizer,
    # Vision
    images_list: List[str],
    num_image_token: int,  # tokens/image
    # Regression params
    pattern_endpoints_list: List[torch.Tensor],
    pattern_transf_list: List[torch.Tensor],
):
    """Tokenize conversations and assemble endpoint/transform embedding inputs.

    Returns:
        The conversations, token ids, attention masks, and padded
        endpoint/transformation tensors with their masks.
    """
    # Tokenize input
    processed_conversation_list = deepcopy(conversation_list)
    processed_conversation_list = replace_placeholder(
        conversation_list=processed_conversation_list,
        images_list=images_list,
        patterns_list=pattern_list,
        num_image_token=num_image_token,
    )
    token_ids = tokenize_conversation(processed_conversation_list, tokenizer)
    attention_masks = token_ids.ne(tokenizer.pad_token_id)

    # Preprocess params and structure them
    # to be used for positional embedding
    pattern_endpoints, pattern_endpoint_masks = preprocess_embedding_params(pattern_endpoints_list)
    pattern_transfs, transf_masks = preprocess_embedding_params(pattern_transf_list)

    return (
        # processed_conversation_list is not used after this point
        conversation_list,
        token_ids,
        attention_masks,
        pattern_endpoints,
        pattern_endpoint_masks,
        pattern_transfs,
        transf_masks,
    )


def prepare_labels(
    input_ids: torch.Tensor,
    question_attention_masks: torch.Tensor,
    attention_masks: torch.Tensor,
) -> torch.Tensor:
    """Mask question and trailing tokens so loss applies only to the answer span."""
    labels = input_ids.clone()
    for idx, (question_mask, conv_mask) in enumerate(
        zip(question_attention_masks, attention_masks)
    ):
        # question is in conv
        question_len = question_mask.sum().item()
        conv_len = conv_mask.sum().item()
        labels[idx, :question_len] = IGNORE_TOKEN_ID
        # Ignore the last \n and padding
        labels[idx, conv_len - 1 :] = IGNORE_TOKEN_ID

    return labels


def assert_single_turn_conv(
    conversation_list,
    conv_type="internvl2_5",
):
    """
    Assert that conversation_list contains single-turn conversations.
    """
    conv = conversation_lib[conv_type]
    for conversation in conversation_list:
        if conv_type == "internvl2_5":
            sep = conv.sep + conv.roles[1]
        else:
            raise NotImplementedError(f"Conversation type {conv_type} is not implemented.")

        n_turns = conversation.count(sep)
        if n_turns != 1:
            raise ValueError(
                f"Conversation `{conversation}` is not a single-turn conversation. "
                f"Got: {n_turns} turns, expected 1 turn. Multiple turns are not supported."
            )


def prepare_params_to_predict(
    question_conv_list: List[str],
    conversation_list: List[str],
    pattern_param_list: List[List[Dict[str, torch.Tensor]]],
):
    """Collect answer-only pattern parameters into padded per-key target tensors.

    Returns:
        The padded parameter targets and their validity masks, keyed by parameter.
    """
    # Prepare params to predict
    # # List[Dict[token_id, torch.Tensor]]: Each dict is a sample.
    param_targets_per_sample: List[Dict[str, torch.Tensor]] = []
    for idx, (question, conv) in enumerate(zip(question_conv_list, conversation_list)):
        n_question_patterns = question.count(DEFAULT_PLACEHOLDER_TOKEN)
        n_conv_patterns = conv.count(DEFAULT_PLACEHOLDER_TOKEN)

        assert n_question_patterns < n_conv_patterns, (
            "Full conversation should contain more patterns than question."
            f" Got {n_conv_patterns} patterns in conversation and "
            f"{n_question_patterns} patterns in question.\n"
        )

        param_target = defaultdict(list)
        for param_dict in pattern_param_list[idx][n_question_patterns:]:
            for k, v in param_dict.items():
                param_target[k].append(v)
        param_target = {k: torch.cat(v) for k, v in param_target.items()}
        param_targets_per_sample.append(param_target)
    # # List[Dict] -> Dict[List]
    param_targets_dict = defaultdict(list)
    param_targets_mask_dict = defaultdict(list)
    all_keys = set([k for param_target in param_targets_per_sample for k in param_target.keys()])
    for sample in param_targets_per_sample:
        for k in all_keys:
            v = sample.get(k, None)
            mask = None if v is None else torch.ones(v.shape[:-1], dtype=torch.bool)
            param_targets_dict[k].append(v)
            param_targets_mask_dict[k].append(mask)

    def _get_shape(params_list):
        """Return the common tensor shape across non-None params, asserting consistency."""
        shape = None
        for param in params_list:
            if param is None:
                continue
            if shape is not None:
                assert shape[1:] == param.shape[1:], (
                    f"Shape mismatch: expected (*,{shape}), " f"got (*,{param.shape[1:]})."
                )
            else:
                shape = param.shape
        assert shape is not None, (
            "No parameters found to predict. " "Make sure that the pattern_param_list is not empty."
        )
        return shape

    def _fill_none(params_list, shape, dtype):
        """Replace None entries with zero tensors of the given shape and dtype."""
        placeholder = torch.zeros(shape, dtype=dtype)
        for idx in range(len(params_list)):
            if params_list[idx] is None:
                params_list[idx] = placeholder
        return params_list

    for k in all_keys:
        shape = _get_shape(param_targets_dict[k])
        param_targets_dict[k] = _fill_none(param_targets_dict[k], shape, dtype=torch.float32)
        param_targets_dict[k] = torch.nn.utils.rnn.pad_sequence(
            param_targets_dict[k], batch_first=True, padding_value=0.0
        )

        # exclude information dim (last dim)
        param_targets_mask_dict[k] = _fill_none(
            param_targets_mask_dict[k], shape[:-1], dtype=torch.bool
        )
        param_targets_mask_dict[k] = torch.nn.utils.rnn.pad_sequence(
            param_targets_mask_dict[k], batch_first=True, padding_value=False
        )
    return (
        param_targets_dict,
        param_targets_mask_dict,
    )


def collate_fn(
    batch,
    *,
    num_image_token: int,  # tokens/image
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    conv_type="internvl2_5",
    generation_only=False,
):
    """Collate raw samples into the tensor dict consumed by the model.

    When ``generation_only`` is set, only question-side inputs are prepared;
    otherwise labels and parameter targets for the full conversation are added.
    """
    (
        sample_type_list,
        # answer
        encoded_pattern_list,
        pattern_param_list,
        pattern_endpoints_list,
        pattern_transf_list,
        # question
        question_pattern_list,
        question_pattern_param_list,
        question_pattern_endpoints_list,
        question_pattern_transf_list,
        # other
        image_path_list,
        images_list,
        gt_pattern_list,
        conversation_list,
        question_conv_list,
        questions_list,
        offset_list,
    ) = _extract_batch(batch)

    assert_single_turn_conv(conversation_list, conv_type=conv_type)
    assert_single_turn_conv(question_conv_list, conv_type=conv_type)

    (
        question_conv_list,
        questions_ids,
        question_attention_masks,
        question_pattern_endpoints,
        question_pattern_endpoint_masks,
        question_pattern_transfs,
        question_pattern_transf_masks,
    ) = prepare_conversation(  # Only questions preparation
        conversation_list=question_conv_list,
        pattern_list=question_pattern_list,
        tokenizer=tokenizer,
        images_list=images_list,
        num_image_token=num_image_token,
        pattern_endpoints_list=question_pattern_endpoints_list,
        pattern_transf_list=question_pattern_transf_list,
    )

    if not generation_only:  # == Ignore this step if inference
        (
            conversation_list,
            input_ids,
            attention_masks,
            pattern_endpoints,
            pattern_endpoint_masks,
            pattern_transfs,
            pattern_transf_masks,
        ) = prepare_conversation(  # Full conversation preparation
            conversation_list=conversation_list,
            pattern_list=encoded_pattern_list,
            tokenizer=tokenizer,
            images_list=images_list,
            num_image_token=num_image_token,
            pattern_endpoints_list=pattern_endpoints_list,
            pattern_transf_list=pattern_transf_list,
        )

        # Prepare labels
        targets = prepare_labels(
            input_ids=input_ids,
            question_attention_masks=question_attention_masks,
            attention_masks=attention_masks,
        )

        # Prepare params to predict
        param_targets, param_target_masks = prepare_params_to_predict(
            question_conv_list=question_conv_list,
            conversation_list=conversation_list,
            pattern_param_list=pattern_param_list,
        )

    all_imgs = [imgs for imgs in images_list if imgs is not None]
    images_clip = (
        torch.cat(all_imgs, dim=0)
        if all_imgs
        else torch.zeros((1, 3, 448, 448), dtype=torch.float32)
    )  # Default shape for CLIP

    out_dict = {
        "input_len": questions_ids.shape[1],
        "image_paths": image_path_list,
        "sample_type": torch.LongTensor(sample_type_list),
        "pixel_values": images_clip,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "questions_pattern_endpoints": question_pattern_endpoints,
        "questions_pattern_endpoints_mask": question_pattern_endpoint_masks,
        "questions_pattern_transformations": question_pattern_transfs,
        "questions_pattern_transformations_mask": question_pattern_transf_masks,
        "question_ids": questions_ids,
        "question_attention_masks": question_attention_masks,
        "conversation_list": conversation_list,
    }
    if not generation_only:
        out_dict.update(
            {
                "input_len": input_ids.shape[1],  # full conversation instead
                "param_targets": param_targets,
                "param_target_masks": param_target_masks,
                "param_target_endpoints": pattern_endpoints,
                "param_target_endpoints_mask": pattern_endpoint_masks,
                "param_target_transformations": pattern_transfs,
                "param_target_transformations_mask": pattern_transf_masks,
                "gt_patterns": gt_pattern_list,
                "attention_mask": attention_masks,  # no 's' for the key
                "input_ids": input_ids,
                "labels": targets,
            }
        )
    return out_dict
