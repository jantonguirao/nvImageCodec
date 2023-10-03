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

class NvJpegLosslessDecoderPlugin
{
  public:
    explicit NvJpegLosslessDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecDecoderDesc_t* getDecoderDesc();
    static bool isPlatformSupported();

  private:
    struct DecodeState
    {
        DecodeState(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvjpegHandle_t handle,
            nvjpegDevAllocatorV2_t* device_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, int num_threads);
        ~DecodeState();

        struct Sample
        {
            nvimgcodecCodeStreamDesc_t* code_stream;
            nvimgcodecImageDesc_t* image;
            const nvimgcodecDecodeParams_t* params;
        };

        const char* plugin_id_;
        const nvimgcodecFrameworkDesc_t* framework_;
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
        explicit ParseState(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvjpegHandle_t handle);
        ~ParseState();

        const char* plugin_id_;
        const nvimgcodecFrameworkDesc_t* framework_;
        std::vector<unsigned char> buffer_;
        nvjpegJpegStream_t nvjpeg_stream_;
    };

    struct Decoder
    {
        Decoder(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
            const char* options = nullptr);
        ~Decoder();

        nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t* code_stream,
            nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params);
        nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
            nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
        nvimgcodecStatus_t decodeBatch(
            nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

        static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);
        static nvimgcodecStatus_t static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
            nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
        static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t decoder, nvimgcodecCodeStreamDesc_t** code_streams,
            nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

        const char* plugin_id_;
        nvjpegHandle_t handle_;
        nvjpegDevAllocatorV2_t device_allocator_;
        nvjpegPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcodecFrameworkDesc_t* framework_;
        std::unique_ptr<DecodeState> decode_state_batch_;
        std::unique_ptr<ParseState> parse_state_;
        const nvimgcodecExecutionParams_t* exec_params_;

        struct CanDecodeCtx {
            Decoder *this_ptr;
            nvimgcodecProcessingStatus_t* status;
            nvimgcodecCodeStreamDesc_t** code_streams;
            nvimgcodecImageDesc_t** images;
            const nvimgcodecDecodeParams_t* params;
            int num_samples;
            int num_blocks;
            std::vector<std::promise<void>> promise;
        };
    };

    nvimgcodecStatus_t create(
        nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    static nvimgcodecStatus_t static_create(
        void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvjpeg_lossless_decoder";
    nvimgcodecDecoderDesc_t decoder_desc_;
    const nvimgcodecFrameworkDesc_t* framework_;
};

} // namespace nvjpeg
