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
#include "iimage_encoder.h"

namespace nvimgcdcs {

class IEncodeState;
class IImage;
class ICodeStream;

class ImageEncoder : public IImageEncoder
{
  public:
    ImageEncoder(const nvimgcdcsEncoderDesc_t desc, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options);
    ~ImageEncoder() override;
    nvimgcdcsBackendKind_t getBackendKind() const override;
    std::unique_ptr<IEncodeState> createEncodeStateBatch() const override;
    void canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcdcsEncodeParams_t* params,
        std::vector<bool>* result, std::vector<nvimgcdcsProcessingStatus_t>* status) const override;
    std::unique_ptr<ProcessingResultsFuture> encode(IEncodeState* encode_state_batch,
        const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
        const nvimgcdcsEncodeParams_t* params) override;

  private:
    const nvimgcdcsEncoderDesc_t encoder_desc_;
    nvimgcdcsEncoder_t encoder_;
};

} // namespace nvimgcdcs