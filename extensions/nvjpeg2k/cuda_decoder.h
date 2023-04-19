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
#include <nvjpeg2k.h>

namespace nvjpeg2k {

class NvJpeg2kDecoderPlugin
{
  public:
    explicit NvJpeg2kDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework);
    nvimgcdcsDecoderDesc_t getDecoderDesc();

  private:
    struct Decoder;
    struct DecodeState
    {
        explicit DecodeState(Decoder* decoder);
        ~DecodeState();

        nvjpeg2kDecodeState_t handle_;
        Decoder* decoder_;
        cudaStream_t stream_;
        cudaEvent_t event_;
    };

    struct ParseState
    {
        explicit ParseState();
        ~ParseState();

        nvjpeg2kStream_t nvjpeg2k_stream_;
        std::vector<unsigned char> buffer_;
    };

    struct Decoder
    {
        Decoder(const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id);
        ~Decoder();

        nvimgcdcsStatus_t getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size);
        nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_streams,
            nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        nvimgcdcsStatus_t decode(DecodeState* decode_state, ParseState* parse_state, nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageDesc_t image,
            const nvimgcdcsDecodeParams_t* params);

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsDecoder_t decoder);
        static nvimgcdcsStatus_t static_get_capabilities(
            nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size);
        static nvimgcdcsStatus_t static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
            nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);
        static nvimgcdcsStatus_t static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
            nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);

        //TODO this is temporary solution and should be changed to per thread resources
        //similarly as it is in nvjpeg decoder
        DecodeState* getSampleDecodeState(int sample_idx);
        ParseState* getSampleParseState(int sample_idx);

        const std::vector<nvimgcdcsCapability_t>& capabilities_;
        nvjpeg2kHandle_t handle_;
        nvjpeg2kDeviceAllocatorV2_t device_allocator_;
        nvjpeg2kPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcdcsFrameworkDesc_t framework_;
        std::vector<std::unique_ptr<ParseState>> per_sample_parse_state_;
        std::vector<std::unique_ptr<DecodeState>> per_sample_decode_state_;
        int device_id_;
    };

    nvimgcdcsStatus_t create(nvimgcdcsDecoder_t* decoder, int device_id);

    static nvimgcdcsStatus_t static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id);

    struct nvimgcdcsDecoderDesc decoder_desc_;
    std::vector<nvimgcdcsCapability_t> capabilities_;
    const nvimgcdcsFrameworkDesc_t framework_;
};

} // namespace nvjpeg2k
