# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Coordinate encodings (sinusoidal and discrete-bin) for pattern parameters."""

from typing import List, Literal, Optional

import torch
import torch.nn as nn


class BaseEncoding(nn.Module):
    """Abstract base for input coordinate encodings."""

    def __init__(
        self,
        in_dim: int,
        include_input: bool = True,
    ) -> None:
        """Store the input dimension and whether the raw input is appended."""
        super().__init__()
        self.include_input = include_input
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Return the encoding's output dimension."""
        pass

    def forward(
        self,
        in_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the input tensor."""
        pass


class SinusoidalEncoding(BaseEncoding):
    """Multi-scale sinusoidal (NeRF-style) positional encoding.

    Each axis is encoded with frequencies from ``2**min_freq_exp`` to
    ``2**max_freq_exp``.
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = True,
    ) -> None:
        """Store the frequency count and exponent range."""
        super().__init__(in_dim, include_input)
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp

    def get_out_dim(self) -> int:
        """Return the encoded dimension given the frequency count and input flag.

        Raises:
            ValueError: If the input dimension has not been set.
        """
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        if self.num_frequencies == 0:
            return self.in_dim
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the sinusoidal encoding of the input.

        Args:
            in_tensor: Input values, ideally in ``[0, 1]``.

        Returns:
            Encoded values in ``[-1, 1]``, with the raw input appended when
            ``include_input``.
        """
        if self.num_frequencies == 0:
            return in_tensor if self.include_input else torch.zeros_like(in_tensor)
        dtype = in_tensor.dtype
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(
            self.min_freq, self.max_freq, self.num_frequencies, device=in_tensor.device
        )
        # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_in_tensor[..., None] * freqs
        # [..., "input_dim" * "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)

        encoded_inputs = torch.sin(
            torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)
        ).to(dtype)
        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs


class DiscreteEncoding(BaseEncoding):
    """Encode each input axis by binning it and looking up a learned embedding."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bin_num: int,
        max_bounds: List[float],
        min_bounds: List[float],
    ) -> None:
        """Set up per-axis bins and the bin embedding table."""
        super().__init__(in_dim, False)
        self.bin_num = bin_num
        self.bounds = min_bounds + max_bounds
        assert len(self.bounds) == in_dim * 2
        self.out_dim = out_dim
        self.embedding_proj = nn.Embedding(bin_num * in_dim, out_dim)

    def get_out_dim(self) -> int:
        """Return the embedding output dimension."""
        return self.out_dim

    def forward(self, in_tensor: torch.Tensor):
        """Bin each axis within its bounds and return the mean bin embedding."""
        assert in_tensor.shape[-1] == self.in_dim
        bounds = torch.tensor(self.bounds).to(in_tensor)
        ids = (
            (in_tensor - bounds[: self.in_dim])
            / (bounds[self.in_dim :] - bounds[: self.in_dim])
            * (self.bin_num - 1)
        )
        ids = ids.long()
        ids = ids.clamp(0, self.bin_num - 1) + torch.arange(self.in_dim).to(ids) * self.bin_num
        embedding = self.embedding_proj(ids).mean(-2)
        return embedding
