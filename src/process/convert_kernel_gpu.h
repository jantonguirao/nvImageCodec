
// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "nvimgcodec.h"
#include <cuda_runtime_api.h>

namespace nvimgcodec {

void LaunchConvertNormKernel(void* output, nvimgcodecSampleDataType_t output_dtype, const void* input,
    nvimgcodecSampleDataType_t input_dtype, int64_t sz, cudaStream_t stream, int input_precision = 0);

}  // namespace nvimgcodec