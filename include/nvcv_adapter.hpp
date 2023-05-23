/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <map>
#include <tuple>

#include <nvimgcodecs.h>
#include <nvcv/IImage.hpp>
#include <nvcv/IImageData.hpp>
#include <nvcv/ImageFormat.hpp>

namespace nvimgcdcs { namespace adapter { namespace nvcv { namespace {

#define CHECK_NVCV(call)                                     \
    {                                                        \
        NVCVStatus _e = (call);                              \
        if (_e != NVCV_SUCCESS) {                            \
            std::stringstream _error;                        \
            _error << "NVCV Types failure: '#" << _e << "'"; \
            throw std::runtime_error(_error.str());          \
        }                                                    \
    }

constexpr auto ext2loc_buffer_kind(NVCVImageBufferType in_kind)
{
    switch (in_kind) {
    case NVCV_IMAGE_BUFFER_NONE:
        return NVIMGCDCS_IMAGE_BUFFER_KIND_UNSUPPORTED;
    case NVCV_IMAGE_BUFFER_STRIDED_CUDA:
        return NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    case NVCV_IMAGE_BUFFER_STRIDED_HOST:
        return NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;
    case NVCV_IMAGE_BUFFER_CUDA_ARRAY:
        return NVIMGCDCS_IMAGE_BUFFER_KIND_UNSUPPORTED;
    default:
        return NVIMGCDCS_IMAGE_BUFFER_KIND_UNSUPPORTED;
    }
}

constexpr auto ext2loc_color_spec(NVCVColorSpec in)
{
    switch (in) {
    case NVCV_COLOR_SPEC_UNDEFINED:
        return NVIMGCDCS_COLORSPEC_UNSUPPORTED;
    case NVCV_COLOR_SPEC_sRGB:
        return NVIMGCDCS_COLORSPEC_SRGB;
    //case :
    //    return NVIMGCDCS_COLORSPEC_GRAY; //TODO
    case NVCV_COLOR_SPEC_sYCC:
        return NVIMGCDCS_COLORSPEC_SYCC;
    //case :
    //    return NVIMGCDCS_COLORSPEC_CMYK; //TODO
    //case :
    //    return NVIMGCDCS_COLORSPEC_YCCK; //TODO
    default:
        return NVIMGCDCS_COLORSPEC_UNSUPPORTED;
    }
};

constexpr auto ext2loc_css(NVCVChromaSubsampling in)
{
    switch (in) {
    case NVCV_CSS_444:
        return NVIMGCDCS_SAMPLING_444;
    case NVCV_CSS_422:
        return NVIMGCDCS_SAMPLING_422;
    case NVCV_CSS_420:
        return NVIMGCDCS_SAMPLING_420;
    //case :
    //    return NVIMGCDCS_SAMPLING_440; //TODO
    case NVCV_CSS_411:
        return NVIMGCDCS_SAMPLING_411;
    //case :
    //    return NVIMGCDCS_SAMPLING_410; //TODO
    //case :
    //    return NVIMGCDCS_SAMPLING_GRAY; //TODO
    //case :
    //   return NVIMGCDCS_SAMPLING_410V; //TODO
    default:
        return NVIMGCDCS_SAMPLING_UNSUPPORTED;
    }
}

constexpr auto ext2loc_sample_type(NVCVDataKind data_kind, int32_t bpp)
{
    switch (data_kind) {
    case NVCV_DATA_KIND_SIGNED: 
        if (bpp <= 8)
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT8;
        else if ((bpp > 8) &&  (bpp <= 16))
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT16;
        else if ((bpp > 16) && (bpp <= 32))
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT32;
        else if ((bpp > 32) && (bpp <= 64))
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT64;
        else
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED;
    
    case NVCV_DATA_KIND_UNSIGNED: 
        if (bpp <= 8)
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        else if ((bpp > 8) && (bpp <= 16))
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
        else if ((bpp > 16) && (bpp <= 32))
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT32;
        else if ((bpp > 32) && (bpp <= 64))
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT64;
        else
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED;
    
    case NVCV_DATA_KIND_FLOAT:
        if (bpp <= 16)
            return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT16;
        else if ((bpp > 16) && (bpp <= 32))
            return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32;
        else if ((bpp > 32) && (bpp <= 64))
            return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT64;
        else
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED;

    case NVCV_DATA_KIND_COMPLEX:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED;
    default:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED;
    }
}

constexpr auto ext2loc_sample_type(NVCVDataType data_type)
{
    switch (data_type) {
    case NVCV_DATA_TYPE_S8:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_INT8;
    case NVCV_DATA_TYPE_U8:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
    case NVCV_DATA_TYPE_S16:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_INT16;
    case NVCV_DATA_TYPE_U16:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
    case NVCV_DATA_TYPE_S32:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_INT32;
    case NVCV_DATA_TYPE_U32:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT32;
    case NVCV_DATA_TYPE_S64:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_INT64;
    case NVCV_DATA_TYPE_U64:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT64;
    case NVCV_DATA_TYPE_F16:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT16;
    case NVCV_DATA_TYPE_F32:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32;
    case NVCV_DATA_TYPE_F64:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT64;
    default:
        return NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED;
    }
}

constexpr auto loc2ext_dtype(nvimgcdcsSampleDataType_t in)
{
    switch (in) {
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UNKNOWN:
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED:
        return NVCV_DATA_TYPE_NONE;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT8:
        return NVCV_DATA_TYPE_S8;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8:
        return NVCV_DATA_TYPE_U8;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT16:
        return NVCV_DATA_TYPE_S16;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16:
        return NVCV_DATA_TYPE_U16;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT32:
        return NVCV_DATA_TYPE_S32;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT32:
        return NVCV_DATA_TYPE_U32;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT64:
        return NVCV_DATA_TYPE_S64;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT64:
        return NVCV_DATA_TYPE_U64;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT16:
        return NVCV_DATA_TYPE_F16;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32:
        return NVCV_DATA_TYPE_F32;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT64:
        return NVCV_DATA_TYPE_F64;
    default:
        return NVCV_DATA_TYPE_NONE;
    }
}

constexpr auto ext2loc_sample_format(int32_t num_planes, NVCVSwizzle swizzle, NVCVColorSpec color_spec)
{
    if (color_spec == NVCV_COLOR_SPEC_sRGB) {
        if (swizzle == NVCV_SWIZZLE_XYZ0)
            return num_planes > 1 ? NVIMGCDCS_SAMPLEFORMAT_P_RGB : NVIMGCDCS_SAMPLEFORMAT_I_RGB;
        else if (swizzle == NVCV_SWIZZLE_ZYX0)
            return num_planes > 1 ? NVIMGCDCS_SAMPLEFORMAT_P_BGR : NVIMGCDCS_SAMPLEFORMAT_I_BGR;
        else
            return NVIMGCDCS_SAMPLEFORMAT_UNSUPPORTED;
    } else if (color_spec == NVCV_COLOR_SPEC_sYCC) {
        if (swizzle == NVCV_SWIZZLE_XYZ0)
            return num_planes > 1 ? NVIMGCDCS_SAMPLEFORMAT_P_YUV : NVIMGCDCS_SAMPLEFORMAT_P_Y;
        else
            return NVIMGCDCS_SAMPLEFORMAT_UNSUPPORTED;
    } else {
        if (swizzle == NVCV_DETAIL_MAKE_SWZL(1, 1, 1, 1)) //TODO confirm this
            return num_planes > 1 ? NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED : NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED;
        else
            return NVIMGCDCS_SAMPLEFORMAT_UNSUPPORTED;
    }
}

constexpr auto loc2ext_buffer_kind(nvimgcdcsImageBufferKind_t in_kind)
{
    switch (in_kind) {
    case NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST:
        return NVCV_IMAGE_BUFFER_STRIDED_HOST;
    case NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE:
        return NVCV_IMAGE_BUFFER_STRIDED_CUDA;
    default:
        return NVCV_IMAGE_BUFFER_NONE;
    }
}

constexpr auto loc2ext_css(nvimgcdcsChromaSubsampling_t in)
{
    switch (in) {
    case NVIMGCDCS_SAMPLING_444:
        return NVCV_CSS_444;
    case NVIMGCDCS_SAMPLING_422:
        return NVCV_CSS_422;
    case NVIMGCDCS_SAMPLING_420:
        return NVCV_CSS_420;
    case NVIMGCDCS_SAMPLING_440:
        return NVCV_CSS_NONE; //TODO
    case NVIMGCDCS_SAMPLING_411:
        return NVCV_CSS_411;
    case NVIMGCDCS_SAMPLING_410:
        return NVCV_CSS_NONE; //TODO
    case NVIMGCDCS_SAMPLING_GRAY:
        return NVCV_CSS_NONE; //TODO
    case NVIMGCDCS_SAMPLING_410V:
        return NVCV_CSS_NONE; //TODO
    default:
        return NVCV_CSS_NONE;
    }
}

constexpr auto loc2ext_color_spec(nvimgcdcsColorSpec_t in)
{
    switch (in) {
    case NVIMGCDCS_COLORSPEC_UNKNOWN:
        return NVCV_COLOR_SPEC_UNDEFINED;
    case NVIMGCDCS_COLORSPEC_SRGB:
        return NVCV_COLOR_SPEC_sRGB;
    case NVIMGCDCS_COLORSPEC_GRAY:
        return NVCV_COLOR_SPEC_UNDEFINED; //TODO
    case NVIMGCDCS_COLORSPEC_SYCC:
        return NVCV_COLOR_SPEC_sYCC;
    case NVIMGCDCS_COLORSPEC_CMYK:
        return NVCV_COLOR_SPEC_UNDEFINED; //TODO
    case NVIMGCDCS_COLORSPEC_YCCK:
        return NVCV_COLOR_SPEC_UNDEFINED; //TODO
    case NVIMGCDCS_COLORSPEC_UNSUPPORTED:
        return NVCV_COLOR_SPEC_UNDEFINED;
    default:
        return NVCV_COLOR_SPEC_UNDEFINED;
    }
}

constexpr auto loc2ext_color_model(nvimgcdcsColorSpec_t in)
{
    switch (in) {
    case NVIMGCDCS_COLORSPEC_UNKNOWN:
        return NVCV_COLOR_MODEL_UNDEFINED;
    case NVIMGCDCS_COLORSPEC_SRGB:
        return NVCV_COLOR_MODEL_RGB;
    case NVIMGCDCS_COLORSPEC_GRAY:
        return NVCV_COLOR_MODEL_UNDEFINED; //TODO
    case NVIMGCDCS_COLORSPEC_SYCC:
        return NVCV_COLOR_MODEL_YCbCr;
    case NVIMGCDCS_COLORSPEC_CMYK:
        return NVCV_COLOR_MODEL_UNDEFINED; //TODO
    case NVIMGCDCS_COLORSPEC_YCCK:
        return NVCV_COLOR_MODEL_UNDEFINED; //TODO
    case NVIMGCDCS_COLORSPEC_UNSUPPORTED:
        return NVCV_COLOR_MODEL_UNDEFINED;
    default:
        return NVCV_COLOR_MODEL_UNDEFINED;
    }
}

constexpr auto loc2ext_data_kind(nvimgcdcsSampleDataType_t in)
{
    switch (in) {
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UNKNOWN:
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED:
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8:
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16:
        return NVCV_DATA_KIND_UNSIGNED;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT8:
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT16:
        return NVCV_DATA_KIND_SIGNED;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32:
        return NVCV_DATA_KIND_FLOAT;
    default:
        return NVCV_DATA_KIND_UNSIGNED;
    }
}

constexpr auto loc2ext_swizzle(nvimgcdcsSampleFormat_t in)
{
    switch (in) {
    case NVIMGCDCS_SAMPLEFORMAT_UNKNOWN:
    case NVIMGCDCS_SAMPLEFORMAT_UNSUPPORTED:
        return NVCV_SWIZZLE_0000;
    case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED:
        return NVCV_SWIZZLE_XYZW;
    case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED:
        return NVCV_SWIZZLE_XYZW;
    case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
        return NVCV_SWIZZLE_XYZ0;
    case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
        return NVCV_SWIZZLE_XYZ0;
    case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
        return NVCV_SWIZZLE_ZYX0;
    case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
        return NVCV_SWIZZLE_ZYX0;
    case NVIMGCDCS_SAMPLEFORMAT_P_Y:
        return NVCV_SWIZZLE_X000;
    case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
        return NVCV_SWIZZLE_XYZ0;
    default:
        return NVCV_SWIZZLE_0000;
    }
}

constexpr unsigned char loc2ext_bpp(nvimgcdcsSampleDataType_t in)
{
    switch (in) {
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UNKNOWN:
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED:
        return 0;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT8:
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8:
        return 8;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT16:
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16:
        return 16;
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32:
        return 32;
    default:
        return 0;
    }
}

constexpr auto loc2ext_packing(const nvimgcdcsImageInfo_t& image_info)
{
    switch (image_info.sample_format) {
    case NVIMGCDCS_SAMPLEFORMAT_UNKNOWN:
    case NVIMGCDCS_SAMPLEFORMAT_UNSUPPORTED:
        return std::make_tuple(NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED: {
        NVCVPacking packing0 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[0].sample_type), 1));
        return std::make_tuple<NVCVPacking, NVCVPacking, NVCVPacking, NVCVPacking>(
            std::move(packing0), NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    }
    case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED: {
        NVCVPacking packing0 = static_cast<NVCVPacking>(
            NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[0].sample_type) * image_info.plane_info[0].num_channels,
                image_info.plane_info[0].num_channels));
        return std::make_tuple<NVCVPacking, NVCVPacking, NVCVPacking, NVCVPacking>(
            std::move(packing0), NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    }
    case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
    case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
    case NVIMGCDCS_SAMPLEFORMAT_P_RGB: {
        NVCVPacking packing0 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[0].sample_type), 1));
        NVCVPacking packing1 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[1].sample_type), 1));
        NVCVPacking packing2 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[2].sample_type), 1));
        return std::make_tuple<NVCVPacking, NVCVPacking, NVCVPacking, NVCVPacking>(
            std::move(packing0), std::move(packing1), std::move(packing2), NVCV_PACKING_0);
    }
    case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
    case NVIMGCDCS_SAMPLEFORMAT_I_BGR: {
        NVCVPacking packing0 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[0].sample_type) * 3, 3));
        return std::make_tuple<NVCVPacking, NVCVPacking, NVCVPacking, NVCVPacking>(
            std::move(packing0), NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    }
    case NVIMGCDCS_SAMPLEFORMAT_P_Y: {
        NVCVPacking packing0 = static_cast<NVCVPacking>(NVCV_DETAIL_BPP_NCH(loc2ext_bpp(image_info.plane_info[0].sample_type), 1));
        return std::make_tuple<NVCVPacking, NVCVPacking, NVCVPacking, NVCVPacking>(
            std::move(packing0), NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    }
    default:
        return std::make_tuple(NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0);
    }
}

nvimgcdcsStatus_t ImageData2Imageinfo(nvimgcdcsImageInfo_t* image_info, const NVCVImageData& image_data)
{
    try {
        image_info->buffer_kind = ext2loc_buffer_kind(image_data.bufferType);
        if (image_info->buffer_kind == NVIMGCDCS_IMAGE_BUFFER_KIND_UNSUPPORTED) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }

        NVCVDataKind data_kind;
        CHECK_NVCV(nvcvImageFormatGetDataKind(image_data.format, &data_kind));

        NVCVSwizzle swizzle;
        CHECK_NVCV(nvcvImageFormatGetSwizzle(image_data.format, &swizzle));
        if (swizzle == NVCV_SWIZZLE_0000)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;

        const NVCVImageBufferStrided& strided = image_data.buffer.strided;
        image_info->num_planes = strided.numPlanes;
        auto ptr = strided.planes[0].basePtr;
        if (ptr == nullptr)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        image_info->buffer = reinterpret_cast<void*>(ptr);

        for (int32_t p = 0; p < strided.numPlanes; ++p) {
            if (strided.planes[p].basePtr != ptr) //Accept only contignous memory
                return NVIMGCDCS_STATUS_INVALID_PARAMETER;
            image_info->plane_info[p].width = strided.planes[p].width;
            image_info->plane_info[p].height = strided.planes[p].height;
            image_info->plane_info[p].row_stride = strided.planes[p].rowStride;
            int32_t bpp;
            CHECK_NVCV(nvcvImageFormatGetPlaneBitsPerPixel(image_data.format, p, &bpp));
            image_info->plane_info[p].sample_type = ext2loc_sample_type(data_kind, bpp);
            image_info->plane_info[p].precision = bpp;
            if (image_info->plane_info[p].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED)
                return NVIMGCDCS_STATUS_INVALID_PARAMETER;
            int32_t num_channels;
            CHECK_NVCV(nvcvImageFormatGetPlaneNumChannels(image_data.format, p, &num_channels));
            image_info->plane_info[p].num_channels = num_channels;
            ptr += image_info->plane_info[p].height * image_info->plane_info[p].row_stride;
        }

        NVCVColorSpec color_spec;
        CHECK_NVCV(nvcvImageFormatGetColorSpec(image_data.format, &color_spec));
        image_info->color_spec = ext2loc_color_spec(color_spec);
        if (image_info->color_spec == NVIMGCDCS_COLORSPEC_UNSUPPORTED)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;

        NVCVChromaSubsampling css;
        CHECK_NVCV(nvcvImageFormatGetChromaSubsampling(image_data.format, &css));
        image_info->chroma_subsampling = ext2loc_css(css);
        if (image_info->chroma_subsampling == NVIMGCDCS_SAMPLING_UNSUPPORTED)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;

        image_info->sample_format = ext2loc_sample_format(strided.numPlanes, swizzle, color_spec);
        if (image_info->sample_format == NVIMGCDCS_SAMPLEFORMAT_UNSUPPORTED)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t ImageInfo2ImageData(NVCVImageData* image_data, const nvimgcdcsImageInfo_t& image_info)
{
    image_data->bufferType = loc2ext_buffer_kind(image_info.buffer_kind);
    if (image_data->bufferType == NVCV_IMAGE_BUFFER_NONE) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    NVCVImageBufferStrided& strided = image_data->buffer.strided;
    strided.numPlanes = image_info.num_planes;
    NVCVByte* ptr = reinterpret_cast<NVCVByte*>(image_info.buffer);
    for (int32_t p = 0; p < strided.numPlanes; ++p) {
        strided.planes[p].width = image_info.plane_info[p].width;
        strided.planes[p].height = image_info.plane_info[p].height;
        strided.planes[p].rowStride = image_info.plane_info[p].row_stride;
        strided.planes[p].basePtr = ptr;
        ptr += image_info.plane_info[p].height * image_info.plane_info[p].row_stride;
    }

    auto color_spec = loc2ext_color_spec(image_info.color_spec);
    if (color_spec == NVCV_COLOR_SPEC_UNDEFINED)
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    auto css = loc2ext_css(image_info.chroma_subsampling);
    auto color_model = loc2ext_color_model(image_info.color_spec);
    if (color_model == NVCV_COLOR_MODEL_UNDEFINED)
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    auto data_kind = loc2ext_data_kind(image_info.plane_info[0].sample_type);
    auto swizzle = loc2ext_swizzle(image_info.sample_format);
    if (swizzle == NVCV_SWIZZLE_0000)
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    auto [packing0, packing1, packing2, packing3] = loc2ext_packing(image_info);
    if (packing0 == NVCV_PACKING_0 && packing1 == NVCV_PACKING_0 && packing2 == NVCV_PACKING_0 && packing3 == NVCV_PACKING_0)
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    image_data->format = NVCV_DETAIL_MAKE_FMTTYPE(
        color_model, color_spec, css, NVCV_MEM_LAYOUT_PITCH_LINEAR, data_kind, swizzle, packing0, packing1, packing2, packing3);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t TensorData2ImageInfo(nvimgcdcsImageInfo_t* image_info, const NVCVTensorData& tensor_data)
{
    try {
        if (tensor_data.bufferType != NVCV_TENSOR_BUFFER_STRIDED_CUDA)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        if (tensor_data.rank > 4)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;

        const NVCVTensorBufferStrided& strided = tensor_data.buffer.strided;
        if (strided.basePtr == nullptr)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;

        image_info->buffer = static_cast<void*>(strided.basePtr);
        image_info->buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        image_info->color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        image_info->chroma_subsampling = NVIMGCDCS_SAMPLING_444;
        auto sample_type = ext2loc_sample_type(tensor_data.dtype);
        if (nvcvTensorLayoutCompare(tensor_data.layout, NVCV_TENSOR_NHWC) == 0 && tensor_data.shape[3] == 3) {
            image_info->sample_format = NVIMGCDCS_SAMPLEFORMAT_I_RGB;
            image_info->plane_info[0].height = tensor_data.shape[1];
            image_info->plane_info[0].width = tensor_data.shape[2];
            image_info->plane_info[0].row_stride = tensor_data.shape[2] * strided.strides[2];
            image_info->plane_info[0].sample_type = sample_type;
        } else if (nvcvTensorLayoutCompare(tensor_data.layout, NVCV_TENSOR_HWC) == 0 && tensor_data.shape[2] == 3) {
            image_info->sample_format = NVIMGCDCS_SAMPLEFORMAT_I_RGB;
            image_info->plane_info[0].height = tensor_data.shape[0];
            image_info->plane_info[0].width = tensor_data.shape[1];
            image_info->plane_info[0].row_stride = tensor_data.shape[1] * strided.strides[1];
            image_info->plane_info[0].sample_type = sample_type;
        } else if (nvcvTensorLayoutCompare(tensor_data.layout, NVCV_TENSOR_NCHW) == 0 && tensor_data.shape[1] == 3) {
            image_info->sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
            image_info->plane_info[0].height = tensor_data.shape[2];
            image_info->plane_info[0].width = tensor_data.shape[3];
            image_info->plane_info[0].row_stride = tensor_data.shape[3] * strided.strides[3];
        } else if (nvcvTensorLayoutCompare(tensor_data.layout, NVCV_TENSOR_CHW) == 0 && tensor_data.shape[0] == 3) {
            image_info->sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
            image_info->plane_info[0].height = tensor_data.shape[1];
            image_info->plane_info[0].width = tensor_data.shape[2];
            image_info->plane_info[0].row_stride = tensor_data.shape[2] * strided.strides[2];
        } else
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;

        if (image_info->sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB) {
            image_info->num_planes = 1;
            image_info->plane_info[0].num_channels = 3;
        } else if (image_info->sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) {
            image_info->num_planes = 3;
            for (auto p = 0; p < image_info->num_planes; ++p) {
                image_info->plane_info[p].height = image_info->plane_info[0].height;
                image_info->plane_info[p].width = image_info->plane_info[0].width;
                image_info->plane_info[p].row_stride = image_info->plane_info[0].row_stride;
                image_info->plane_info[p].num_channels = 1;
                image_info->plane_info[p].sample_type = sample_type;
            }
        }
        image_info->buffer_size = image_info->plane_info[0].row_stride * image_info->plane_info[0].height * image_info->num_planes;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t ImageInfo2TensorData(NVCVTensorData* tensor_data, const nvimgcdcsImageInfo_t& image_info)
{
    try {
        if (image_info.buffer_kind != NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        tensor_data->bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
        NVCVTensorBufferStrided& strided = tensor_data->buffer.strided;
        strided.basePtr = static_cast<NVCVByte*>(image_info.buffer);
        tensor_data->rank = 4;
        tensor_data->dtype = loc2ext_dtype(image_info.plane_info[0].sample_type);
        if (tensor_data->dtype == NVCV_DATA_TYPE_NONE)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        int32_t bpp;
        CHECK_NVCV(nvcvDataTypeGetBitsPerPixel(tensor_data->dtype, &bpp));
        int32_t bytes = (bpp + 7) / 8;
        if (tensor_data->dtype == NVCV_DATA_TYPE_NONE)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) {
            tensor_data->layout = NVCV_TENSOR_NCHW;
            tensor_data->shape[0] = 1;
            tensor_data->shape[1] = 3;
            tensor_data->shape[2] = image_info.plane_info[0].height;
            tensor_data->shape[3] = image_info.plane_info[0].width;

            strided.strides[3] = bytes;
            strided.strides[2] = image_info.plane_info[0].row_stride;
            strided.strides[1] = strided.strides[2] * tensor_data->shape[2];
            strided.strides[0] = strided.strides[1] * tensor_data->shape[1];
        } else if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB) {
            tensor_data->layout = NVCV_TENSOR_NHWC;
            tensor_data->shape[0] = 1;
            tensor_data->shape[1] = image_info.plane_info[0].height;
            tensor_data->shape[2] = image_info.plane_info[0].width;
            tensor_data->shape[3] = 3;

            strided.strides[3] = bytes;
            strided.strides[2] = strided.strides[3] * tensor_data->shape[3];
            strided.strides[1] = image_info.plane_info[0].row_stride;
            strided.strides[0] = strided.strides[1] * tensor_data->shape[1];
        } else {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

}}}} // namespace nvimgcdcs::adapter::nvcv
