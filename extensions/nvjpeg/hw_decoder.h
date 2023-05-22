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

class NvJpegHwDecoderPlugin
{
  public:
    explicit NvJpegHwDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework);
    nvimgcdcsDecoderDesc_t getDecoderDesc();
    static bool isPlatformSupported();

  private:
    struct DecodeState
    {
        DecodeState(
            nvjpegHandle_t handle, nvjpegDevAllocatorV2_t* device_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, int num_threads);
        ~DecodeState();

        struct Sample
        {
            nvimgcdcsCodeStreamDesc_t code_stream;
            nvimgcdcsImageDesc_t image;
            const nvimgcdcsDecodeParams_t* params;
        };

        nvjpegHandle_t handle_;
        nvjpegJpegState_t state_;
        cudaStream_t stream_;
        cudaEvent_t event_;
        nvjpegDevAllocatorV2_t* device_allocator_;
        nvjpegPinnedAllocatorV2_t* pinned_allocator_;

        std::vector<Sample> samples_;
    };

    struct ParseState
    {
        explicit ParseState(nvjpegHandle_t handle);
        ~ParseState();
        std::vector<unsigned char> buffer_;
        nvjpegJpegStream_t nvjpeg_stream_;
    };

    struct Decoder
    {
        Decoder(const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id);
        ~Decoder();

        nvimgcdcsStatus_t getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size);
        nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvjpegHandle_t handle, nvimgcdcsCodeStreamDesc_t* code_streams,
            nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        nvimgcdcsStatus_t decodeBatch();

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsDecoder_t decoder);
        static nvimgcdcsStatus_t static_get_capabilities(
            nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size);
        static nvimgcdcsStatus_t static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
            nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        static nvimgcdcsStatus_t static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
            nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);

        const std::vector<nvimgcdcsCapability_t>& capabilities_;
        nvjpegHandle_t handle_;

        nvjpegDevAllocatorV2_t device_allocator_;
        nvjpegPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcdcsFrameworkDesc_t framework_;
        std::unique_ptr<DecodeState> decode_state_batch_;
        std::unique_ptr<ParseState> parse_state_;
        int device_id_;
    };

    nvimgcdcsStatus_t create(nvimgcdcsDecoder_t* decoder, int device_id, const char* options);

    static nvimgcdcsStatus_t static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id, const char* options);

    struct nvimgcdcsDecoderDesc decoder_desc_;
    std::vector<nvimgcdcsCapability_t> capabilities_;
    const nvimgcdcsFrameworkDesc_t framework_;
};

} // namespace nvjpeg
