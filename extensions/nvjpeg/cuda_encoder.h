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
#include <nvjpeg.h>
#include <future>
#include <vector>

namespace nvjpeg {

class NvJpegCudaEncoderPlugin
{
  public:
    explicit NvJpegCudaEncoderPlugin(const nvimgcdcsFrameworkDesc_t* framework);
    nvimgcdcsEncoderDesc_t* getEncoderDesc();

  private:
    struct Encoder;
    struct EncodeState
    {
        explicit EncodeState(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, nvjpegHandle_t handle, int num_threads);
        ~EncodeState();

        struct PerThreadResources
        {
            cudaStream_t stream_;
            cudaEvent_t event_;
            nvjpegEncoderState_t state_;
            std::vector<unsigned char> compressed_data_; //TODO it should be created with pinned allocator
        };

        struct Sample
        {
            nvimgcdcsCodeStreamDesc_t* code_stream_;
            nvimgcdcsImageDesc_t* image_;
            const nvimgcdcsEncodeParams_t* params;
        };
        
        const char* plugin_id_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        nvjpegHandle_t handle_;
        std::vector<PerThreadResources> per_thread_;
        std::vector<Sample> samples_;
    };

    struct Encoder
    {
        Encoder(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, const nvimgcdcsExecutionParams_t* exec_params, const char* options);
        ~Encoder();

        
        nvimgcdcsStatus_t canEncode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t** images,
            nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);
        nvimgcdcsStatus_t encode(int sample_idx);
        nvimgcdcsStatus_t encodeBatch(
            nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsEncoder_t encoder);
        static nvimgcdcsStatus_t static_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
            nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);
        static nvimgcdcsStatus_t static_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t** images,
            nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);

        const char* plugin_id_;
        nvjpegHandle_t handle_;
        nvjpegDevAllocatorV2_t device_allocator_;
        nvjpegPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        std::unique_ptr<EncodeState> encode_state_batch_;
        const nvimgcdcsExecutionParams_t* exec_params_;
        std::string options_;
    };

    nvimgcdcsStatus_t create(
        nvimgcdcsEncoder_t* encoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);
    static nvimgcdcsStatus_t static_create(
        void* instance, nvimgcdcsEncoder_t* encoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvjpeg_cuda_encoder";
    nvimgcdcsEncoderDesc_t encoder_desc_;
    const nvimgcdcsFrameworkDesc_t* framework_;
};

} // namespace nvjpeg
