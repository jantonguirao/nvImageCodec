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

nvjpegOutputFormat_t nvimgcodec_to_nvjpeg_format(nvimgcodecSampleFormat_t nvimgcodec_format);
nvjpegExifOrientation_t nvimgcodec_to_nvjpeg_orientation(nvimgcodecOrientation_t orientation);
nvjpegChromaSubsampling_t nvimgcodec_to_nvjpeg_css(nvimgcodecChromaSubsampling_t nvimgcodec_css);
nvjpegJpegEncoding_t nvimgcodec_to_nvjpeg_encoding(nvimgcodecJpegEncoding_t nvimgcodec_encoding);

nvimgcodecSampleDataType_t precision_to_sample_type(int precision);
nvimgcodecChromaSubsampling_t nvjpeg_to_nvimgcodec_css(nvjpegChromaSubsampling_t nvjpeg_css);
nvimgcodecOrientation_t exif_to_nvimgcodec_orientation(nvjpegExifOrientation_t exif_orientation);
nvimgcodecJpegEncoding_t nvjpeg_to_nvimgcodec_encoding(nvjpegJpegEncoding_t nvjpeg_encoding);
