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
#include <vector>
#include "processing_results.h"

namespace nvimgcdcs {
class IDecodeState;
class IImage;
class ICodeStream;

class IImageDecoder
{
  public:
    virtual ~IImageDecoder() = default;
    virtual std::unique_ptr<IDecodeState> createDecodeState() const = 0;
    virtual std::unique_ptr<IDecodeState> createDecodeStateBatch() const = 0;
    virtual void getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size) = 0;
    virtual bool canDecode(nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageDesc_t image,
       const  nvimgcdcsDecodeParams_t* params) const = 0;
    virtual void canDecode(const std::vector<ICodeStream*>& code_streams,
        const std::vector<IImage*>& images, const nvimgcdcsDecodeParams_t* params,
        std::vector<bool>* result) const = 0;
    virtual std::unique_ptr<ProcessingResultsFuture> decode(
        ICodeStream* code_stream, IImage* image, const nvimgcdcsDecodeParams_t* params) = 0;
    virtual std::unique_ptr<ProcessingResultsFuture> decodeBatch(IDecodeState* decode_state_batch,
        const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
        const nvimgcdcsDecodeParams_t* params) = 0;
};

} // namespace nvimgcdcs