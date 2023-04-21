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
#include <future>
#include <vector>
#include <nvjpeg.h>

namespace nvjpeg {

class NvJpegCudaEncoderPlugin
{
  public:
    explicit NvJpegCudaEncoderPlugin(const nvimgcdcsFrameworkDesc_t framework);
    nvimgcdcsEncoderDesc_t getEncoderDesc();

  private:
    struct Encoder;
    struct EncodeState
    {
        explicit EncodeState(Encoder* encoder);
        ~EncodeState();

        nvjpegEncoderState_t handle_ = nullptr;
        std::vector<unsigned char> compressed_data_; //TODO it should be created with pinned allocator
        nvimgcdcsImageDesc_t image_;
        nvimgcdcsCodeStreamDesc_t code_stream_;
        Encoder* encoder_;
        cudaStream_t stream_;
        cudaEvent_t event_;

        //TODO this is temporary solution and should be changed to per thread resources similarly as it is in nvjpeg decoder
        EncodeState* getSampleEncodeState(int sample_idx);
        std::vector<std::unique_ptr<EncodeState>> per_sample_encode_state_;
    };

    struct Encoder
    {
        Encoder(const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework,
            int device_id);
        ~Encoder();

        nvimgcdcsStatus_t getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size);
        nvimgcdcsStatus_t canEncode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t* images,
            nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);
        nvimgcdcsStatus_t encode(NvJpegCudaEncoderPlugin::EncodeState* encode_state, nvimgcdcsImageDesc_t image,
            nvimgcdcsCodeStreamDesc_t code_stream, const nvimgcdcsEncodeParams_t* params);

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsEncoder_t encoder);
        static nvimgcdcsStatus_t static_get_capabilities(
            nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size);
        static nvimgcdcsStatus_t static_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
            nvimgcdcsImageDesc_t* images, nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);
        static nvimgcdcsStatus_t static_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t* images,
            nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);

        const std::vector<nvimgcdcsCapability_t>& capabilities_;
        nvjpegHandle_t handle_;
        nvjpegDevAllocatorV2_t device_allocator_;
        nvjpegPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcdcsFrameworkDesc_t framework_;
        std::unique_ptr<EncodeState> encode_state_batch_;
        int device_id_;
    };

    nvimgcdcsStatus_t create(nvimgcdcsEncoder_t* encoder, int device_id);
    static nvimgcdcsStatus_t static_create(void* instance, nvimgcdcsEncoder_t* encoder, int device_id);

    struct nvimgcdcsEncoderDesc encoder_desc_;
    std::vector<nvimgcdcsCapability_t> capabilities_;
    const nvimgcdcsFrameworkDesc_t framework_;
};

} // namespace nvjpeg