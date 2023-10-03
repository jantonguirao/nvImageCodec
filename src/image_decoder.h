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

#include <nvimgcodec.h>
#include <memory>
#include <string>
#include <thread>
#include "iimage_decoder.h"

namespace nvimgcodec {
class IDecodeState;
class IImage;
class ICodeStream;

class ImageDecoder : public IImageDecoder
{
  public:
    ImageDecoder(const nvimgcodecDecoderDesc_t* desc, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    ~ImageDecoder() override;
    nvimgcodecBackendKind_t getBackendKind() const override;
    std::unique_ptr<IDecodeState> createDecodeStateBatch() const override;
    void canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcodecDecodeParams_t* params,
        std::vector<bool>* result, std::vector<nvimgcodecProcessingStatus_t>* status) const override;
    std::unique_ptr<ProcessingResultsFuture> decode(IDecodeState* decode_state_batch,
        const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
        const nvimgcodecDecodeParams_t* params) override;
    const char* decoderId() const override;

  private:
    const nvimgcodecDecoderDesc_t* decoder_desc_;
    nvimgcodecDecoder_t decoder_;
};

} // namespace nvimgcodec