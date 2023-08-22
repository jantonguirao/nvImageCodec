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
    explicit NvJpegHwDecoderPlugin(const nvimgcdcsFrameworkDesc_t* framework);
    nvimgcdcsDecoderDesc_t* getDecoderDesc();
    static bool isPlatformSupported();

  private:
    struct DecodeState
    {
        DecodeState(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework,
            nvjpegHandle_t handle, nvjpegDevAllocatorV2_t* device_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, int num_threads);
        ~DecodeState();

        struct Sample
        {
            nvimgcdcsCodeStreamDesc_t* code_stream;
            nvimgcdcsImageDesc_t* image;
            const nvimgcdcsDecodeParams_t* params;
        };

        const char* plugin_id_;
        const nvimgcdcsFrameworkDesc_t* framework_;
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
        explicit ParseState(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, nvjpegHandle_t handle);
        ~ParseState();

        const char* plugin_id_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        std::vector<unsigned char> buffer_;
        nvjpegJpegStream_t nvjpeg_stream_;
    };

    struct Decoder
    {
        Decoder(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, const nvimgcdcsExecutionParams_t* exec_params,
            const char* options = nullptr);
        ~Decoder();

        nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_stream,
            nvimgcdcsImageDesc_t* image, const nvimgcdcsDecodeParams_t* params, ParseState &parse_state);
        nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t** code_streams,
            nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        nvimgcdcsStatus_t decodeBatch(
            nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsDecoder_t decoder);
        static nvimgcdcsStatus_t static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
            nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        static nvimgcdcsStatus_t static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t** code_streams,
            nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);

        void parseOptions(const char* options);

        const char* plugin_id_;
        nvjpegHandle_t handle_;
        nvjpegDevAllocatorV2_t device_allocator_;
        nvjpegPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        std::unique_ptr<DecodeState> decode_state_batch_;
        std::vector<std::unique_ptr<ParseState>> parse_state_;
        const nvimgcdcsExecutionParams_t* exec_params_;
        unsigned int num_hw_engines_;
        unsigned int num_cores_per_hw_engine_;
        float hw_load_ = 1.0f;

        int preallocate_batch_size_ = 1;
        int preallocate_width_ = 1;
        int preallocate_height_ = 1;

        struct CanDecodeCtx {
            Decoder *this_ptr;
            nvimgcdcsProcessingStatus_t* status;
            nvimgcdcsCodeStreamDesc_t** code_streams;
            nvimgcdcsImageDesc_t** images;
            const nvimgcdcsDecodeParams_t* params;
            int num_samples;
            int num_blocks;
            std::vector<std::promise<void>> promise;
        };

    };

    nvimgcdcsStatus_t create(
        nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

    static nvimgcdcsStatus_t static_create(
        void* instance, nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvjpeg_hw_decoder";
    nvimgcdcsDecoderDesc_t decoder_desc_;
    const nvimgcdcsFrameworkDesc_t* framework_;
};

} // namespace nvjpeg
