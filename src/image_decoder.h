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
#include <memory>
#include <string>
#include <thread>
#include "iimage_decoder.h"

namespace nvimgcdcs {
class IDecodeState;
class IImage;
class ICodeStream;

class ImageDecoder : public IImageDecoder
{
  public:
    ImageDecoder(const nvimgcdcsDecoderDesc_t desc, const nvimgcdcsDecodeParams_t* params);
    ~ImageDecoder() override;
    std::unique_ptr<IDecodeState> createDecodeState() const override;
    std::unique_ptr<IDecodeState> createDecodeStateBatch() const override;
    void getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size) override;
    void canDecode(const std::vector<ICodeStream*>& code_streams,
        const std::vector<IImage*>& images, const nvimgcdcsDecodeParams_t* params,
        std::vector<bool>* result) const override;
    std::unique_ptr<ProcessingResultsFuture> decode(IDecodeState* decode_state_batch,
        const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
        const nvimgcdcsDecodeParams_t* params) override;

  private:
    const nvimgcdcsDecoderDesc_t decoder_desc_;
    nvimgcdcsDecoder_t decoder_;
};

} // namespace nvimgcdcs