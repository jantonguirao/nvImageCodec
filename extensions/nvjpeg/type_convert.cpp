/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "type_convert.h"

nvjpegOutputFormat_t nvimgcdcs_to_nvjpeg_format(nvimgcdcsSampleFormat_t nvimgcdcs_format)
{
    switch (nvimgcdcs_format) {
    case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED:
        return NVJPEG_OUTPUT_UNCHANGED;
    case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED:
        return NVJPEG_OUTPUT_UNCHANGED;
    case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
        return NVJPEG_OUTPUT_RGB;
    case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
        return NVJPEG_OUTPUT_RGBI;
    case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
        return NVJPEG_OUTPUT_BGR;
    case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
        return NVJPEG_OUTPUT_BGRI;
    case NVIMGCDCS_SAMPLEFORMAT_P_Y:
        return NVJPEG_OUTPUT_Y;
    case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
        return NVJPEG_OUTPUT_YUV;
    default:
        return NVJPEG_OUTPUT_UNCHANGED;
    }
}

nvjpegExifOrientation_t nvimgcdcs_to_nvjpeg_orientation(nvimgcdcsOrientation_t orientation)
{
    if (orientation.rotated == 0 && !orientation.flip_x && !orientation.flip_y) {
        return NVJPEG_ORIENTATION_NORMAL;
    } else if (orientation.rotated == 0 && orientation.flip_x && !orientation.flip_y) {
        return NVJPEG_ORIENTATION_FLIP_HORIZONTAL;
    } else if (orientation.rotated == 180 && !orientation.flip_x && !orientation.flip_y) {
        return NVJPEG_ORIENTATION_ROTATE_180;
    } else if (orientation.rotated == 0 && !orientation.flip_x && orientation.flip_y) {
        return NVJPEG_ORIENTATION_FLIP_VERTICAL;
    } else if (orientation.rotated == 90 && !orientation.flip_x && orientation.flip_y) {
        return NVJPEG_ORIENTATION_TRANSPOSE;
    } else if (orientation.rotated == 270 && !orientation.flip_x && !orientation.flip_y) {
        return NVJPEG_ORIENTATION_ROTATE_90;
    } else if (orientation.rotated == 270 && !orientation.flip_x && orientation.flip_y) {
        return NVJPEG_ORIENTATION_TRANSVERSE;
    } else if (orientation.rotated == 90 && !orientation.flip_x && !orientation.flip_y) {
        return NVJPEG_ORIENTATION_ROTATE_270;
    } else {
        return NVJPEG_ORIENTATION_UNKNOWN;
    }
}

nvjpegChromaSubsampling_t nvimgcdcs_to_nvjpeg_css(nvimgcdcsChromaSubsampling_t nvimgcdcs_css)
{
    switch (nvimgcdcs_css) {
    case NVIMGCDCS_SAMPLING_UNSUPPORTED:
        return NVJPEG_CSS_UNKNOWN;
    case NVIMGCDCS_SAMPLING_444:
        return NVJPEG_CSS_444;
    case NVIMGCDCS_SAMPLING_422:
        return NVJPEG_CSS_422;
    case NVIMGCDCS_SAMPLING_420:
        return NVJPEG_CSS_420;
    case NVIMGCDCS_SAMPLING_440:
        return NVJPEG_CSS_440;
    case NVIMGCDCS_SAMPLING_411:
        return NVJPEG_CSS_411;
    case NVIMGCDCS_SAMPLING_410:
        return NVJPEG_CSS_410;
    case NVIMGCDCS_SAMPLING_GRAY:
        return NVJPEG_CSS_GRAY;
    case NVIMGCDCS_SAMPLING_410V:
        return NVJPEG_CSS_410V;
    default:
        return NVJPEG_CSS_UNKNOWN;
    }
}

nvjpegJpegEncoding_t nvimgcdcs_to_nvjpeg_encoding(nvimgcdcsJpegEncoding_t nvimgcdcs_encoding)
{
    switch (nvimgcdcs_encoding) {
    case NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT:
        return NVJPEG_ENCODING_BASELINE_DCT;
    case NVIMGCDCS_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN:
        return NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN;
    case NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN:
        return NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;
    default:
        return NVJPEG_ENCODING_UNKNOWN;
    }
}

nvimgcdcsJpegEncoding_t nvjpeg_to_nvimgcdcs_encoding(nvjpegJpegEncoding_t nvjpeg_encoding)
{
    switch (nvjpeg_encoding) {
    case NVJPEG_ENCODING_BASELINE_DCT:
        return NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT;
    case NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN:
        return NVIMGCDCS_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN;
    case NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN:
        return NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;
    default:
        return NVIMGCDCS_JPEG_ENCODING_UNKNOWN;
    }
}

nvimgcdcsSampleDataType_t precision_to_sample_type(int precision)
{
    return precision == 8 ? NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 : NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED;
}

nvimgcdcsChromaSubsampling_t nvjpeg_to_nvimgcdcs_css(nvjpegChromaSubsampling_t nvjpeg_css)
{
    switch (nvjpeg_css) {
    case NVJPEG_CSS_UNKNOWN:
        return NVIMGCDCS_SAMPLING_NONE;
    case NVJPEG_CSS_444:
        return NVIMGCDCS_SAMPLING_444;
    case NVJPEG_CSS_422:
        return NVIMGCDCS_SAMPLING_422;
    case NVJPEG_CSS_420:
        return NVIMGCDCS_SAMPLING_420;
    case NVJPEG_CSS_440:
        return NVIMGCDCS_SAMPLING_440;
    case NVJPEG_CSS_411:
        return NVIMGCDCS_SAMPLING_411;
    case NVJPEG_CSS_410:
        return NVIMGCDCS_SAMPLING_410;
    case NVJPEG_CSS_GRAY:
        return NVIMGCDCS_SAMPLING_GRAY;
    case NVJPEG_CSS_410V:
        return NVIMGCDCS_SAMPLING_410V;
    default:
        return NVIMGCDCS_SAMPLING_UNSUPPORTED;
    }
}

nvimgcdcsOrientation_t exif_to_nvimgcdcs_orientation(nvjpegExifOrientation_t exif_orientation)
{
    switch (exif_orientation) {
    case NVJPEG_ORIENTATION_NORMAL:
        return {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
    case NVJPEG_ORIENTATION_FLIP_HORIZONTAL:
        return {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, true, false};
    case NVJPEG_ORIENTATION_ROTATE_180:
        return {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 180, false, false};
    case NVJPEG_ORIENTATION_FLIP_VERTICAL:
        return {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, true};
    case NVJPEG_ORIENTATION_TRANSPOSE:
        return {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 90, false, true};
    case NVJPEG_ORIENTATION_ROTATE_90:
        return {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 270, false, false};
    case NVJPEG_ORIENTATION_TRANSVERSE:
        return {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 270, false, true};
    case NVJPEG_ORIENTATION_ROTATE_270:
        return {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 90, false, false};
    default:
        return {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
    }
}
