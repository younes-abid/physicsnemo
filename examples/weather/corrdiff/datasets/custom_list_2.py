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
import time as time_pkg
import math
from typing import List, Tuple, Union

import json
import numpy as np
from numba import jit, prange
import xarray as xr

from physicsnemo.utils.diffusion import convert_datetime_to_cftime

from datasets.base import ChannelMetadata, DownscalingDataset
import gc

class CustomDataset(DownscalingDataset):
    """Reader for custom dataset used for CorrDiff with support for multiple preloaded files."""

    def __init__(
        self,
        data_path: List[str],  # List of file paths
        stats_path: str,
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        invariant_variables: Union[List[str], None] = None,  # Add this argument
    ):
        # Ignore invariant_variables since your dataset does not use it
        del invariant_variables  # Optional: explicitly discard it

        self.file_paths = data_path
        self.stats_path = stats_path
        self.input_variables = input_variables
        self.output_variables = output_variables

        # Track file sizes and cumulative sizes
        self.file_sizes = []
        self.cumulative_sizes = []
        total_size = 0
        for file_path in self.file_paths:
            with xr.open_dataset(file_path, group="input") as ds:
                size = ds.sizes["sample"]
                self.file_sizes.append(size)
                total_size += size
                self.cumulative_sizes.append(total_size)

        self.total_size = total_size

        # Load normalization stats
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.input_mean, self.input_std = _load_stats(stats, input_variables, "input")
        self.output_mean, self.output_std = _load_stats(stats, output_variables, "output")

        # Initialize lists to store preloaded data
        self.inputs = []
        self.outputs = []
        self.times = []
        self.coords = []

        # Preload all files
        for file_idx in range(len(self.file_paths)):
            input_data, output_data, time_data, coord_data, img_shape, upsample_factor = self._load_file(file_idx)
            self.inputs.append(input_data)
            self.outputs.append(output_data)
            self.times.append(time_data)
            self.coords.append(coord_data)

        # Set image shape and upsample factor (assume all files have the same shape)
        self.img_shape = img_shape
        self.upsample_factor = upsample_factor

    def _load_file(self, file_idx):
        """Load the file at the given index into memory and update all attributes."""
        print()
        print(f"Loading file {file_idx + 1}/{len(self.file_paths)}: {self.file_paths[file_idx]}")
        
        file_path = self.file_paths[file_idx]

        # Start timing
        start_time = time_pkg.time()

        # Load input and output data
        input_data, input_variables = _load_dataset(file_path, "input", self.input_variables)
        output_data, output_variables = _load_dataset(file_path, "output", self.output_variables)

        # Load temporal and spatial coordinates
        with xr.open_dataset(file_path) as ds:
            if "time" in ds.variables:
                time_data = np.array(ds["time"])
            else:
                time_data = np.arange(input_data.shape[0])  # Use dummy values if missing
            coord_data = np.array(ds["coord"])

        # Determine image shape and upsample factor
        img_shape = output_data.shape[-2:]
        upsample_factor = output_data.shape[-1] // input_data.shape[-1]

        # End timing
        end_time = time_pkg.time()

        # Log the time taken to load the file
        print(f"Loaded file: {file_path} (Index: {file_idx}) in {end_time - start_time:.2f} seconds")

        return input_data, output_data, time_data, coord_data, img_shape, upsample_factor

    def __getitem__(self, idx):
        """Return the data sample (output, input) at index idx."""
        # Determine which file contains the requested index
        file_idx = next(i for i, size in enumerate(self.cumulative_sizes) if idx < size)

        # Get the relative index within the file
        relative_idx = idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)

        # Retrieve data from the preloaded lists
        x = self.inputs[file_idx][relative_idx]
        y = self.outputs[file_idx][relative_idx]

        # Normalize data
        x = (x - self.input_mean) / self.input_std
        y = (y - self.output_mean) / self.output_std
        return y, x

    def __len__(self):
        """Return the total number of samples across all files."""
        return self.total_size

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
            datetime.datetime.utcfromtimestamp(t.tolist() / 1e9) for t in np.concatenate(self.times)
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