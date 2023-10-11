
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


#include "convert.h"
#include "type_utils.h"
#include "static_switch.h"
#include <stdexcept>

namespace nvimgcodec {

#define IMG_TYPES uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float

template <typename Out, typename In>
__global__ void ConvertNormKernel(Out *output, const In *input, int64_t sz) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sz)
        return;
    output[idx] = ConvertNorm<Out>(input[idx]);
}

template <typename Out, typename In>
__global__ void ConvertNormCustomPrecisionKernel(Out *output, const In *input, int64_t sz, float multiplier) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sz)
        return;
    output[idx] = ConvertNorm<Out, float>(multiplier * input[idx]);
}

template <typename Out, typename In>
void LaunchConvertNormKernelImpl(Out *output, const In *input, int64_t sz, cudaStream_t stream, int input_precision) {
    const unsigned int block = sz < 1024 ? sz : 1024;
    const unsigned int grid = (sz + block - 1) / block;
    constexpr auto input_dtype = type2id<In>::value;
    if (NeedDynamicRangeScaling(input_precision, input_dtype)) {
        float multiplier = DynamicRangeMultiplier(input_precision, input_dtype) / MaxValue(input_dtype);
        ConvertNormCustomPrecisionKernel<Out, In><<<grid, block, 0, stream>>>(output, input, sz, multiplier);
    } else {
        ConvertNormKernel<Out, In><<<grid, block, 0, stream>>>(output, input, sz);
    }
}

void LaunchConvertNormKernel(void* output, nvimgcodecSampleDataType_t output_dtype, const void* input,
    nvimgcodecSampleDataType_t input_dtype, int64_t sz, cudaStream_t stream, int input_precision) {
    TYPE_SWITCH(output_dtype, type2id, Output, (IMG_TYPES),
        (TYPE_SWITCH(input_dtype, type2id, Input, (IMG_TYPES),
            (LaunchConvertNormKernelImpl<Output, Input>(
                reinterpret_cast<Output*>(output), reinterpret_cast<const Input*>(input), sz, stream, input_precision);),
            std::runtime_error("Unsupported input type"))),
        std::runtime_error("Unsupported output type"))
}

}  // namespace nvimgcodec