/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <string>
#include <vector>
#include <tuple>

#include <nvimgcodecs.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "jpeg2k_encode_params.h"
#include "jpeg_encode_params.h"

namespace nvimgcdcs {

namespace py = pybind11;
using namespace py::literals;

class EncodeParams
{
  public:
    EncodeParams();

    float getQuality() { return encode_params_.quality; }
    void setQuality(float quality) { encode_params_.quality = quality; };

    float getTargetPsnr() { return encode_params_.target_psnr; }
    void setTargetPsnr(float target_psnr) { encode_params_.target_psnr = target_psnr; };

    nvimgcdcsColorSpec_t getColorSpec() { return color_spec_; }
    void setColorSpec(nvimgcdcsColorSpec_t color_spec) { color_spec_ = color_spec; };

    nvimgcdcsChromaSubsampling_t getChromaSubsampling() { return chroma_subsampling_; }
    void setChromaSubsampling(nvimgcdcsChromaSubsampling_t chroma_subsampling) { chroma_subsampling_ = chroma_subsampling; }

    Jpeg2kEncodeParams& getJpeg2kEncodeParams() { return jpeg2k_encode_params_; }
    void setJpeg2kEncodeParams(Jpeg2kEncodeParams jpeg2k_encode_params) { jpeg2k_encode_params_ = jpeg2k_encode_params; }

    JpegEncodeParams& getJpegEncodeParams() { return jpeg_encode_params_; }
    void setJpegEncodeParams(JpegEncodeParams jpeg_encode_params) { jpeg_encode_params_ = jpeg_encode_params; }
    static void exportToPython(py::module& m);

    Jpeg2kEncodeParams jpeg2k_encode_params_;
    JpegEncodeParams jpeg_encode_params_;
    nvimgcdcsEncodeParams_t encode_params_;
    nvimgcdcsChromaSubsampling_t chroma_subsampling_;
    nvimgcdcsColorSpec_t color_spec_;
};

} // namespace nvimgcdcs
