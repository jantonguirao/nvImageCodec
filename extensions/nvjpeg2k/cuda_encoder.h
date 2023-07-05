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
#include <vector>
#include "nvjpeg2k.h"

#include <nppdefs.h>

namespace nvjpeg2k {

class NvJpeg2kEncoderPlugin
{
  public:
    explicit NvJpeg2kEncoderPlugin(const nvimgcdcsFrameworkDesc_t framework);
    nvimgcdcsEncoderDesc_t getEncoderDesc();

  private:
    struct Encoder;
    struct EncodeState
    {
        explicit EncodeState(nvjpeg2kEncoder_t handle, nvimgcdcsDeviceAllocator_t* device_allocator,
            nvimgcdcsPinnedAllocator_t* pinned_allocator, int device_id, int num_threads);
        ~EncodeState();

        struct PerThreadResources
        {
            cudaStream_t stream_;
            cudaEvent_t event_;
            nvjpeg2kEncodeState_t state_;
            std::vector<unsigned char> compressed_data_; //TODO it should be created with pinned allocator

            NppStreamContext npp_ctx_;
        };

        struct Sample
        {
            nvimgcdcsCodeStreamDesc_t* code_stream;
            nvimgcdcsImageDesc_t* image;
            const nvimgcdcsEncodeParams_t* params;
        };

        nvjpeg2kEncoder_t handle_;
        nvimgcdcsDeviceAllocator_t* device_allocator_;
        nvimgcdcsPinnedAllocator_t* pinned_allocator_;
        int device_id_;
        std::vector<PerThreadResources> per_thread_;
        std::vector<Sample> samples_;
    };

    struct Encoder
    {
        Encoder(const nvimgcdcsFrameworkDesc_t framework, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options);
        ~Encoder();

        
        nvimgcdcsStatus_t canEncode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t** images,
            nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);
        nvimgcdcsStatus_t encode(int sample_idx);
        nvimgcdcsStatus_t encodeBatch();

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsEncoder_t encoder);
        static nvimgcdcsStatus_t static_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
            nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);
        static nvimgcdcsStatus_t static_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t** images,
            nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);

        
        nvjpeg2kEncoder_t handle_;
        const nvimgcdcsFrameworkDesc_t framework_;
        std::unique_ptr<EncodeState> encode_state_batch_;
        int device_id_;
        const nvimgcdcsBackendParams_t* backend_params_;
        std::string options_;
    };

    nvimgcdcsStatus_t create(
        nvimgcdcsEncoder_t* encoder, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options);
    static nvimgcdcsStatus_t static_create(
        void* instance, nvimgcdcsEncoder_t* encoder, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options);

    struct nvimgcdcsEncoderDesc encoder_desc_;
    
    const nvimgcdcsFrameworkDesc_t framework_;
};

} // namespace nvjpeg2k
