# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import math
from typing import List, Tuple, Union

import json
import numpy as np
from numba import jit, prange
import xarray as xr

from physicsnemo.utils.diffusion import convert_datetime_to_cftime

from datasets.base import ChannelMetadata, DownscalingDataset
from torchvision.transforms import Resize
import torch


class CustomDataset(DownscalingDataset):
    """Reader for custom dataset used for CorrDiff."""

    def __init__(
        self,
        data_path: str,
        stats_path: str,
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        invariant_variables: Union[List[str], None] = None,  # Add this argument
    ):
        # Ignore invariant_variables since your dataset does not use it
        del invariant_variables  # Optional: explicitly discard it

        # Load data
        (self.input, self.input_variables) = _load_dataset(
            data_path, "input", input_variables
        )
        (self.output, self.output_variables) = _load_dataset(
            data_path, "output", output_variables
        )

        # Load temporal and spatial coordinates
        with xr.open_dataset(data_path) as ds:
            #TODO make sure to add variables to the dataset time and a dimension coord similarly to hrrrr data like this 
            # Variables in the root group: ['time']
            # Dimensions in the root group: ['sample', 'coord']
            if "time" in ds.variables:
                self.times = np.array(ds["time"])
            else:
                self.times = np.arange(self.input.shape[0])  # Use dummy values if missing
            self.coords = np.array(ds["coord"])

        self.img_shape = self.output.shape[-2:]
        self.upsample_factor = self.output.shape[-1] // self.input.shape[-1]

        # Load normalization stats
        with open(stats_path, "r") as f:
            stats = json.load(f)
        (self.input_mean, self.input_std) = _load_stats(stats, self.input_variables, "input")
        (self.output_mean, self.output_std) = _load_stats(stats, self.output_variables, "output")

    def __getitem__(self, idx):
        """Return the data sample (output, input) at index idx."""
        x = self.upsample(self.input[idx].copy())
        y = self.output[idx]
        
        print('='*80, flush=True)
        # Log original shapes
        print(f"Original x shape: {x.shape}, Original y shape: {y.shape}", flush=True)
        
        # Convert NumPy arrays to PyTorch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Log tensor shapes before resizing
        print(f"Tensor x shape before resize: {x.shape}, Tensor y shape before resize: {y.shape}", flush=True)
            
        # Resize tensors to 448x448
        resize = Resize((448, 448))
        x = resize(x)
        y = resize(y)
        # Log tensor shapes after resizing
        print(f"Tensor x shape after resize: {x.shape}, Tensor y shape after resize: {y.shape}", flush=True)
        
        # convert back to numpy
        x = x.numpy()
        y = y.numpy()
        
        x = self.normalize_input(x)
        y = self.normalize_output(y)
        
        # Log final shapes
        print(f"Final x shape: {x.shape}, Final y shape: {y.shape}", flush=True)
        return (y, x)

    def __len__(self):
        return self.input.shape[0]

    def longitude(self) -> np.ndarray:
        """Get longitude values from the dataset."""
        return np.full(self.img_shape, np.nan)

    def latitude(self) -> np.ndarray:
        """Get latitude values from the dataset."""
        return np.full(self.img_shape, np.nan)

    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel."""
        return [ChannelMetadata(name=v) for v in self.input_variables]

    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel."""
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self) -> List:
        """Get time values from the dataset."""
        datetimes = (
            datetime.datetime.utcfromtimestamp(t.tolist() / 1e9) for t in self.times
        )
        return [convert_datetime_to_cftime(t) for t in datetimes]

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        return self.img_shape

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return (x - self.input_mean) / self.input_std

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x * self.input_std + self.input_mean

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return (x - self.output_mean) / self.output_std

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x * self.output_std + self.output_mean

    def upsample(self, x):
        """Extend x around edges with linear extrapolation."""
        y_shape = (
            x.shape[0],
            x.shape[1] * self.upsample_factor,
            x.shape[2] * self.upsample_factor,
        )
        y = np.empty(y_shape, dtype=np.float32)
        _zoom_extrapolate(x, y, self.upsample_factor)
        return y


def _load_dataset(data_path, group, variables=None, stack_axis=1):
    with xr.open_dataset(data_path, group=group) as ds:
        if variables is None:
            variables = list(ds.keys())
        data = np.stack([ds[v] for v in variables], axis=stack_axis)
    return (data, variables)


def _load_stats(stats, variables, group):
    mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(
        np.float32
    )
    std = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(
        np.float32
    )
    return (mean, std)


@jit(nopython=True)
def _zoom_extrapolate(x, y, factor):
    """Bilinear zoom with extrapolation.
    Use a numba function here because numpy/scipy options are rather slow.
    """
    s = 1 / factor
    for k in prange(y.shape[0]):
        for iy in range(y.shape[1]):
            ix = (iy + 0.5) * s - 0.5
            ix0 = int(math.floor(ix))
            ix0 = max(0, min(ix0, x.shape[1] - 2))
            ix1 = ix0 + 1
            for jy in range(y.shape[2]):
                jx = (jy + 0.5) * s - 0.5
                jx0 = int(math.floor(jx))
                jx0 = max(0, min(jx0, x.shape[2] - 2))
                jx1 = jx0 + 1

                x00 = x[k, ix0, jx0]
                x01 = x[k, ix0, jx1]
                x10 = x[k, ix1, jx0]
                x11 = x[k, ix1, jx1]
                djx = jx - jx0
                x0 = x00 + djx * (x01 - x00)
                x1 = x10 + djx * (x11 - x10)
                y[k, iy, jy] = x0 + (ix - ix0) * (x1 - x0)