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
#include <nvjpeg.h>
#include <future>
#include <vector>

namespace nvjpeg {

class NvJpegCudaEncoderPlugin
{
  public:
    explicit NvJpegCudaEncoderPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecEncoderDesc_t* getEncoderDesc();

  private:
    struct Encoder;
    struct EncodeState
    {
        explicit EncodeState(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvjpegHandle_t handle, int num_threads);
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
            nvimgcodecCodeStreamDesc_t* code_stream_;
            nvimgcodecImageDesc_t* image_;
            const nvimgcodecEncodeParams_t* params;
        };
        
        const char* plugin_id_;
        const nvimgcodecFrameworkDesc_t* framework_;
        nvjpegHandle_t handle_;
        std::vector<PerThreadResources> per_thread_;
        std::vector<Sample> samples_;
    };

    struct Encoder
    {
        Encoder(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options);
        ~Encoder();

        
        nvimgcodecStatus_t canEncode(nvimgcodecProcessingStatus_t* status, nvimgcodecImageDesc_t** images,
            nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params);
        nvimgcodecStatus_t encode(int sample_idx);
        nvimgcodecStatus_t encodeBatch(
            nvimgcodecImageDesc_t** images, nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params);

        static nvimgcodecStatus_t static_destroy(nvimgcodecEncoder_t encoder);
        static nvimgcodecStatus_t static_can_encode(nvimgcodecEncoder_t encoder, nvimgcodecProcessingStatus_t* status,
            nvimgcodecImageDesc_t** images, nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params);
        static nvimgcodecStatus_t static_encode_batch(nvimgcodecEncoder_t encoder, nvimgcodecImageDesc_t** images,
            nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params);

        const char* plugin_id_;
        nvjpegHandle_t handle_;
        nvjpegDevAllocatorV2_t device_allocator_;
        nvjpegPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcodecFrameworkDesc_t* framework_;
        std::unique_ptr<EncodeState> encode_state_batch_;
        const nvimgcodecExecutionParams_t* exec_params_;
        std::string options_;
    };

    nvimgcodecStatus_t create(
        nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    static nvimgcodecStatus_t static_create(
        void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvjpeg_cuda_encoder";
    nvimgcodecEncoderDesc_t encoder_desc_;
    const nvimgcodecFrameworkDesc_t* framework_;
};

} // namespace nvjpeg
