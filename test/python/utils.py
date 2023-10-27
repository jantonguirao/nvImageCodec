# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import ctypes as c
import os

def get_nvjpeg_ver():
    nvjpeg_ver_major, nvjpeg_ver_minor, nvjpeg_ver_patch = (c.c_int(), c.c_int(), c.c_int())
    try:
        nvjpeg_libname = f'libnvjpeg.so'
        nvjpeg_lib = c.CDLL(nvjpeg_libname)
    except:
        for file in os.listdir("/usr/local/cuda/lib64/"):
            if file.startswith("libnvjpeg.so"):
                nvjpeg_lib = c.CDLL(file)
                nvjpeg_lib.nvjpegGetProperty(0, c.byref(nvjpeg_ver_major))
                nvjpeg_lib.nvjpegGetProperty(1, c.byref(nvjpeg_ver_minor))
                nvjpeg_lib.nvjpegGetProperty(2, c.byref(nvjpeg_ver_patch))
                break
    return nvjpeg_ver_major.value, nvjpeg_ver_minor.value, nvjpeg_ver_patch.value

def get_cuda_compute_capability(device_id=0):
    compute_cap = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_cap = compute_cap[0] + compute_cap[1] / 10.
    except ModuleNotFoundError:
        print("NVML not found")
    return compute_cap