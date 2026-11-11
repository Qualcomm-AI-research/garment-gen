# Copyright (c) 2021 Intel ISL (Intel Intelligent Systems Lab)
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
# Based on: https://github.com/isl-org/DPT/blob/main/dpt/models.py

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from transformers import ViTConfig, ViTModel

from .blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        pred_channels=3,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, pred_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            # x: [bs, seq_len, hidden_size] -> [bs, hidden_size, patch_h, patch_w] (img-like)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            # Linear layer (conv 1x1) - Only change number of channels (aka hidden_size)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out, (int(patch_h * 16), int(patch_w * 16)), mode="bilinear", align_corners=True
        )
        out = self.scratch.output_conv2(out)

        return out


def get_2d_sincos_pos_embed_from_coords(d_model, coords):
    assert coords.shape[1] == 2
    B, _, H, W = coords.shape
    device = coords.device
    omega = torch.arange(d_model // 4, dtype=torch.float32, device=device) / (d_model // 4)
    omega = 1.0 / (10000**omega)  # [d_model//4]

    # coords: [B, 2, H, W] --> x: [B, H, W], y: [B, H, W]
    x = coords[:, 0].reshape(B, -1)  # [B, H*W]
    y = coords[:, 1].reshape(B, -1)  # [B, H*W]

    # y_emb, x_emb: [B, H*W, d_model//4]
    y_emb = y.unsqueeze(-1) * omega  # [B, H*W, d_model//4]
    x_emb = x.unsqueeze(-1) * omega  # [B, H*W, d_model//4]

    pos_emb = torch.cat(
        [torch.sin(y_emb), torch.cos(y_emb), torch.sin(x_emb), torch.cos(x_emb)], dim=-1
    )  # [B, H*W, d_model]

    pos_emb = pos_emb.permute(0, 2, 1).reshape(B, d_model, H, W)  # [B, d_model, H, W]
    return pos_emb


class DPT(nn.Module):
    def __init__(
        self,
        config_path=None,
        pretrained_path=None,
        layers=4,
        features=256,
        pred_channels=3,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        **kwargs,
    ):
        super(DPT, self).__init__()

        if pretrained_path is not None:
            self.pretrained = ViTModel.from_pretrained(pretrained_path)
        else:
            assert config_path is not None
            with open(config_path, "r") as f:
                json_config = json.load(f)

            config = ViTConfig(**json_config)
            self.pretrained = ViTModel(config)

        self.hidden_dim = self.pretrained.config.hidden_size

        self.layers = range(-layers, 0) if isinstance(layers, int) else layers
        self.depth_head = DPTHead(
            self.hidden_dim,
            pred_channels=pred_channels,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
        )

    def forward(self, segmentation_map):
        h, w = segmentation_map.shape[-2:]
        raw_features = self.pretrained(
            pixel_values=segmentation_map, output_hidden_states=True, interpolate_pos_encoding=True
        )
        outputs = []
        cls_tokens = []
        for layer in self.layers:
            outputs.append(raw_features.hidden_states[layer][:, 1:])
            cls_tokens.append(raw_features.hidden_states[layer][:, 0])

        features = tuple(zip(outputs, cls_tokens))

        patch_h, patch_w = h // 16, w // 16

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth
