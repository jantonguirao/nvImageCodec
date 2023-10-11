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
#include "iimage_encoder.h"

namespace nvimgcodec {

class IEncodeState;
class IImage;
class ICodeStream;

class ImageEncoder : public IImageEncoder
{
  public:
    ImageEncoder(const nvimgcodecEncoderDesc_t* desc, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    ~ImageEncoder() override;
    nvimgcodecBackendKind_t getBackendKind() const override;
    std::unique_ptr<IEncodeState> createEncodeStateBatch() const override;
    void canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcodecEncodeParams_t* params,
        std::vector<bool>* result, std::vector<nvimgcodecProcessingStatus_t>* status) const override;
    std::unique_ptr<ProcessingResultsFuture> encode(IEncodeState* encode_state_batch,
        const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
        const nvimgcodecEncodeParams_t* params) override;
    const char* encoderId() const override;

  private:
    const nvimgcodecEncoderDesc_t* encoder_desc_;
    nvimgcodecEncoder_t encoder_;
};

} // namespace nvimgcodec