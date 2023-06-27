/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "lossless_decoder.h"
#include <library_types.h>
#include <nvimgcodecs.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <set>
#include <vector>
#include "errors_handling.h"
#include "log.h"
#include "nvjpeg_utils.h"
#include "type_convert.h"

namespace nvjpeg {

NvJpegLosslessDecoderPlugin::NvJpegLosslessDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL,
          this,                // instance
          "nvjpeg_lossless_decoder", // id
          "jpeg",              // codec_type
          static_create, Decoder::static_destroy, Decoder::static_get_capabilities, Decoder::static_can_decode,
          Decoder::static_decode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT, NVIMGCDCS_CAPABILITY_ROI, NVIMGCDCS_CAPABILITY_LAYOUT_PLANAR,
          NVIMGCDCS_CAPABILITY_LAYOUT_INTERLEAVED}
    , framework_(framework)
{
}

bool NvJpegLosslessDecoderPlugin::isPlatformSupported()
{
    return false;
}

nvimgcdcsDecoderDesc_t NvJpegLosslessDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::canDecode(nvimgcdcsProcessingStatus_t* status, nvjpegHandle_t handle, 
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{        
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

NvJpegLosslessDecoderPlugin::ParseState::ParseState(nvjpegHandle_t handle)
{    
}

NvJpegLosslessDecoderPlugin::ParseState::~ParseState()
{
}


NvJpegLosslessDecoderPlugin::Decoder::Decoder(
    const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id, const char* options)
    : capabilities_(capabilities)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
    , device_id_(device_id)
{    
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::create(nvimgcdcsDecoder_t* decoder, int device_id, const char* options)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id, const char* options)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

NvJpegLosslessDecoderPlugin::Decoder::~Decoder()
{
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_destroy(nvimgcdcsDecoder_t decoder)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_get_capabilities(
    nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    return NVIMGCDCS_STATUS_SUCCESS;
}

NvJpegLosslessDecoderPlugin::DecodeState::DecodeState(
    nvjpegHandle_t handle, nvjpegDevAllocatorV2_t* device_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, int num_threads)
    : handle_(handle)
    , device_allocator_(device_allocator)
    , pinned_allocator_(pinned_allocator)
{    
}

NvJpegLosslessDecoderPlugin::DecodeState::~DecodeState()
{    
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::decodeBatch()
{
    return NVIMGCDCS_STATUS_SUCCESS;
}
nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
    nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{    
    return NVIMGCDCS_STATUS_SUCCESS;
}
} // namespace nvjpeg
