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

nvjpegOutputFormat_t nvimgcdcs_to_nvjpeg_format(nvimgcdcsSampleFormat_t nvimgcdcs_format);
nvjpegExifOrientation_t nvimgcdcs_to_nvjpeg_orientation(nvimgcdcsOrientation_t orientation);
nvjpegChromaSubsampling_t nvimgcdcs_to_nvjpeg_css(nvimgcdcsChromaSubsampling_t nvimgcdcs_css);
nvjpegJpegEncoding_t nvimgcdcs_to_nvjpeg_encoding(nvimgcdcsJpegEncoding_t nvimgcdcs_encoding);

nvimgcdcsSampleDataType_t precision_to_sample_type(int precision);
nvimgcdcsChromaSubsampling_t nvjpeg_to_nvimgcdcs_css(nvjpegChromaSubsampling_t nvjpeg_css);
nvimgcdcsOrientation_t exif_to_nvimgcdcs_orientation(nvjpegExifOrientation_t exif_orientation);
nvimgcdcsJpegEncoding_t nvjpeg_to_nvimgcdcs_encoding(nvjpegJpegEncoding_t nvjpeg_encoding);
