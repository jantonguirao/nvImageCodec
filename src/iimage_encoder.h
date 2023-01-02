/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodecs.h>
#include <nvimgcodecs.h>
#include <memory>
#include <string>

namespace nvimgcdcs {

class IEncodeState;
class IImage;
class ICodeStream;

class IImageEncoder
{
  public:
    virtual ~IImageEncoder() = default;
    virtual std::unique_ptr<IEncodeState> createEncodeState(cudaStream_t cuda_stream) const = 0;
    virtual void getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size) = 0;
    virtual void encode(ICodeStream* code_stream, IImage* image, nvimgcdcsEncodeParams_t* params) = 0;
};

} // namespace nvimgcdcs