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

#include <nppdefs.h>
#include <nvimgcodecs.h>
#include <nvjpeg2k.h>
#include <memory>
#include <vector>
#include <future>

namespace nvjpeg2k {

class NvJpeg2kDecoderPlugin
{
  public:
    explicit NvJpeg2kDecoderPlugin(const nvimgcdcsFrameworkDesc_t* framework);
    nvimgcdcsDecoderDesc_t* getDecoderDesc();

  private:
    struct Decoder;

    struct ParseState
    {
        explicit ParseState(const char* id, const nvimgcdcsFrameworkDesc_t* framework);
        ~ParseState();

        const char* plugin_id_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        nvjpeg2kStream_t nvjpeg2k_stream_;
        std::vector<unsigned char> buffer_;
    };

    struct DecodeState
    {
        explicit DecodeState(const char* id, const nvimgcdcsFrameworkDesc_t* framework, nvjpeg2kHandle_t handle,
            nvimgcdcsDeviceAllocator_t* device_allocator, nvimgcdcsPinnedAllocator_t* pinned_allocator, int device_id, int num_threads);
        ~DecodeState();

        struct PerThreadResources
        {
            cudaStream_t stream_;
            cudaEvent_t event_;
            nvjpeg2kDecodeState_t state_;
            std::unique_ptr<ParseState> parse_state_;

            NppStreamContext npp_ctx_;
        };

        struct Sample
        {
            nvimgcdcsCodeStreamDesc_t* code_stream;
            nvimgcdcsImageDesc_t* image;
            const nvimgcdcsDecodeParams_t* params;
        };

        const char* plugin_id_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        nvjpeg2kHandle_t handle_ = nullptr;
        nvimgcdcsDeviceAllocator_t* device_allocator_;
        nvimgcdcsPinnedAllocator_t* pinned_allocator_;
        int device_id_;
        std::vector<PerThreadResources> per_thread_;
        std::vector<Sample> samples_;
    };

    struct Decoder
    {
        Decoder(const char* id, const nvimgcdcsFrameworkDesc_t* framework, const nvimgcdcsExecutionParams_t* exec_params);
        ~Decoder();

        nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_stream,
            nvimgcdcsImageDesc_t* image, const nvimgcdcsDecodeParams_t* params);
        nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t** code_streams,
            nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        nvimgcdcsStatus_t decode(int sample_idx, bool immediate);
        nvimgcdcsStatus_t decodeBatch(
            nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        nvjpeg2kHandle_t getNvjpeg2kHandle();

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsDecoder_t decoder);
        static nvimgcdcsStatus_t static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
            nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        static nvimgcdcsStatus_t static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t** code_streams,
            nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);

        const char* plugin_id_;
        nvjpeg2kHandle_t handle_;
        nvjpeg2kDeviceAllocatorV2_t device_allocator_;
        nvjpeg2kPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcdcsFrameworkDesc_t* framework_;
        std::unique_ptr<DecodeState> decode_state_batch_;
        const nvimgcdcsExecutionParams_t* exec_params_;

        struct CanDecodeCtx {
            Decoder *this_ptr;
            nvimgcdcsProcessingStatus_t* status;
            nvimgcdcsCodeStreamDesc_t** code_streams;
            nvimgcdcsImageDesc_t** images;
            const nvimgcdcsDecodeParams_t* params;
            int num_samples;
            int num_threads;
            std::vector<std::promise<void>> promise;
        };
    };

    nvimgcdcsStatus_t create(
        nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

    static nvimgcdcsStatus_t static_create(
        void* instance, nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvjpeg2k_decoder";
    nvimgcdcsDecoderDesc_t decoder_desc_;
    const nvimgcdcsFrameworkDesc_t* framework_;
};

} // namespace nvjpeg2k
