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
    virtual nvimgcdcsBackendKind_t getBackendKind() const = 0;
    virtual std::unique_ptr<IDecodeState> createDecodeStateBatch() const = 0;
    virtual void canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
        const nvimgcdcsDecodeParams_t* params, std::vector<bool>* result, std::vector<nvimgcdcsProcessingStatus_t>* status) const = 0;
    virtual std::unique_ptr<ProcessingResultsFuture> decode(IDecodeState* decode_state_batch,
        const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
        const nvimgcdcsDecodeParams_t* params) = 0;
};

} // namespace nvimgcdcs