/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <vector>

namespace nvpnm {

class NvPnmEncoderPlugin
{
  public:
    explicit NvPnmEncoderPlugin(const nvimgcdcsFrameworkDesc_t* framework);
    nvimgcdcsEncoderDesc_t* getEncoderDesc();

  private:
    struct EncodeState
    {
        struct Sample
        {
            nvimgcdcsCodeStreamDesc_t* code_stream;
            nvimgcdcsImageDesc_t* image;
            const nvimgcdcsEncodeParams_t* params;
        };
        const char* plugin_id_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        std::vector<Sample> samples_;
    };

    struct Encoder
    {
        Encoder(
            const char* id, const nvimgcdcsFrameworkDesc_t* framework, const nvimgcdcsBackendParams_t* backend_params, const char* options);
        ~Encoder();

        nvimgcdcsStatus_t canEncode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t** images,
            nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);
        static nvimgcdcsProcessingStatus_t encode(const char* id, const nvimgcdcsFrameworkDesc_t* framework, nvimgcdcsImageDesc_t* image,
            nvimgcdcsCodeStreamDesc_t* code_stream, const nvimgcdcsEncodeParams_t* params);
        nvimgcdcsStatus_t encodeBatch(
            nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsEncoder_t encoder);
        static nvimgcdcsStatus_t static_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
            nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);
        static nvimgcdcsStatus_t static_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t** images,
            nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);

        const char* plugin_id_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        std::unique_ptr<EncodeState> encode_state_batch_;
        const nvimgcdcsBackendParams_t* backend_params_;
        std::string options_;
    };

    nvimgcdcsStatus_t create(
        nvimgcdcsEncoder_t* encoder, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options);
    static nvimgcdcsStatus_t static_create(
        void* instance, nvimgcdcsEncoder_t* encoder, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options);

    static constexpr const char* plugin_id_ = "nvpnm_encoder";
    nvimgcdcsEncoderDesc_t encoder_desc_;
    const nvimgcdcsFrameworkDesc_t* framework_;
};

} // namespace nvpnm
