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

#include <nvimgcdcs_module.h>
#include <nvimgcodecs.h>
#include <memory>
#include <string>

namespace nvimgcdcs {
class IDecodeState;
class IImage;
class ICodeStream;

class IImageDecoder
{
  public:
    virtual ~IImageDecoder() = default;
    virtual std::unique_ptr<IDecodeState> createDecodeState() const = 0;
    virtual void getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size) = 0;
    virtual void decode(ICodeStream* code_stream, IImage* image, nvimgcdcsDecodeParams_t* params) = 0;
};

} // namespace nvimgcdcs