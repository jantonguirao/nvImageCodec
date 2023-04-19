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

class NvJpegCudaDecoderPlugin
{
  public:
    explicit NvJpegCudaDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework);
    nvimgcdcsDecoderDesc_t getDecoderDesc();

  private:
    struct ParseState
    {
        nvjpegJpegStream_t nvjpeg_stream_;
        std::vector<unsigned char> buffer_;
    };

    struct DecodeState
    {
        DecodeState(
            nvjpegHandle_t handle, nvjpegDevAllocatorV2_t* device_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, int num_threads);
        ~DecodeState();

        // Set of resources per-thread.
        // Some of them are double-buffered, so that we can simultaneously decode
        // the host part of the next sample, while the GPU part of the previous
        // is still consuming the data from the previous iteration pinned buffer.
        struct PerThreadResources
        {
            // double-buffered

            struct Page
            {
                struct DecoderData
                {
                    nvjpegJpegDecoder_t decoder = nullptr;
                    nvjpegJpegState_t state = nullptr;
                };
                // indexing via nvjpegBackend_t (NVJPEG_BACKEND_GPU_HYBRID and NVJPEG_BACKEND_HYBRID)
                std::array<DecoderData, 3> decoder_data;
                nvjpegBufferPinned_t pinned_buffer_;
                ParseState parse_state_;
            };
            std::array<Page, 2> pages_;
            int current_page_idx = 0;
            cudaStream_t stream_;
            cudaEvent_t event_;
            nvjpegBufferDevice_t device_buffer_;
        };

        struct Sample
        {
            nvimgcdcsCodeStreamDesc_t code_stream;
            nvimgcdcsImageDesc_t image;
            const nvimgcdcsDecodeParams_t* params;
        };

        struct DecoderData
        {
            nvjpegJpegDecoder_t decoder = nullptr;
            nvjpegJpegState_t state = nullptr;
        };

        nvjpegHandle_t handle_;
        nvjpegDevAllocatorV2_t* device_allocator_;
        nvjpegPinnedAllocatorV2_t* pinned_allocator_;

        std::vector<PerThreadResources> per_thread_;
        std::vector<Sample> samples_;
    };

    struct Decoder
    {
        Decoder(const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id);
        ~Decoder();

        nvimgcdcsStatus_t getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size);
        nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_streams,
            nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        nvimgcdcsStatus_t decode(int sample_idx);
        nvimgcdcsStatus_t decodeBatch();

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsDecoder_t decoder);
        static nvimgcdcsStatus_t static_get_capabilities(
            nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size);
        static nvimgcdcsStatus_t static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_streams,
            nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        static nvimgcdcsStatus_t static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
            nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);

        ParseState* getSampleParseState(int sample_idx);

        const std::vector<nvimgcdcsCapability_t>& capabilities_;
        nvjpegHandle_t handle_;

        nvjpegDevAllocatorV2_t device_allocator_;
        nvjpegPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcdcsFrameworkDesc_t framework_;
        std::unique_ptr<DecodeState> decode_state_batch_;
        int device_id_;
    };

    nvimgcdcsStatus_t create(nvimgcdcsDecoder_t* decoder, int device_id);

    static nvimgcdcsStatus_t static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id);

    struct nvimgcdcsDecoderDesc decoder_desc_;
    std::vector<nvimgcdcsCapability_t> capabilities_;
    const nvimgcdcsFrameworkDesc_t framework_;
};

} // namespace nvjpeg
