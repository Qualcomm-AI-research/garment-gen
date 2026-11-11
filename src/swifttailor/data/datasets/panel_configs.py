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
# Based on: https://github.com/georgeNakayama/AIpparel-Code/blob/master/data/datasets/panel_configs.py

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PanelConfig:
    max_pattern_len: Optional[int]
    max_panel_len: Optional[int]
    max_num_stitches: Optional[int]
    max_stitch_edges: Optional[int]
    panel_classification: Optional[str]
    filter_by_params: Optional[str]


@dataclass
class PanelStats:
    outlines: List[float]
    rotations: List[float]
    stitch_tags: List[float]
    translations: List[float]


@dataclass
class StatsConfig:
    scale: List[float] = field(default_factory=list)
    shift: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"scale": list(self.scale), "shift": list(self.shift)}


@dataclass
class StandardizeConfig:
    outlines: StatsConfig = field(default_factory=StatsConfig)
    rotations: StatsConfig = field(default_factory=StatsConfig)
    stitch_tags: StatsConfig = field(default_factory=StatsConfig)
    translations: StatsConfig = field(default_factory=StatsConfig)
    vertices: StatsConfig = field(default_factory=StatsConfig)

    def to_dict(self) -> dict:
        return {
            "outlines": self.outlines.to_dict(),
            "rotations": self.rotations.to_dict(),
            "stitch_tags": self.stitch_tags.to_dict(),
            "translations": self.translations.to_dict(),
            "vertices": self.vertices.to_dict(),
        }
