/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "encode_params.h"

#include <iostream>
#include <optional>

#include "error_handling.h"

namespace nvimgcdcs {

EncodeParams::EncodeParams()
    : jpeg2k_encode_params_{}
    , jpeg_encode_params_{}
    , encode_params_{NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS, nullptr, 95, 50}
    , chroma_subsampling_{NVIMGCDCS_SAMPLING_444}
    , color_spec_{NVIMGCDCS_COLORSPEC_UNCHANGED}
{
}

void EncodeParams::exportToPython(py::module& m)
{
    py::class_<EncodeParams>(m, "EncodeParams")
        .def(py::init([]() { return EncodeParams{}; }), "Default constructor")
        .def(py::init([](float quality, float target_psnr, nvimgcdcsColorSpec_t color_spec, nvimgcdcsChromaSubsampling_t chroma_subsampling,
                          std::optional<JpegEncodeParams> jpeg_encode_params, std::optional<Jpeg2kEncodeParams> jpeg2k_encode_params) {
            EncodeParams p;
            p.encode_params_.quality = quality;
            p.encode_params_.target_psnr = target_psnr;
            p.color_spec_ = color_spec;
            p.chroma_subsampling_ = chroma_subsampling;
            p.jpeg_encode_params_ = jpeg_encode_params.has_value() ? jpeg_encode_params.value() : JpegEncodeParams();
            p.jpeg2k_encode_params_ = jpeg2k_encode_params.has_value() ? jpeg2k_encode_params.value() : Jpeg2kEncodeParams();

            return p;
        }),
            // clang-format off
            "quality"_a = 95, 
            "target_psnr"_a = 50, 
            "color_spec"_a = NVIMGCDCS_COLORSPEC_UNCHANGED, 
            "chroma_subsampling"_a = NVIMGCDCS_SAMPLING_444,
            "jpeg_encode_params"_a = py::none(),
            "jpeg2k_encode_params"_a = py::none(),
            "Constructor with quality, target_psnr, color_spec, chroma_subsampling etc. parameters")
        // clang-format on
        .def_property("quality", &EncodeParams::getQuality, &EncodeParams::setQuality, "Quality value 0-100 (default 95)")
        .def_property("target_psnr", &EncodeParams::getTargetPsnr, &EncodeParams::setTargetPsnr, "Target psnr (default 50)")
        .def_property("color_spec", &EncodeParams::getColorSpec, &EncodeParams::setColorSpec,
            "Output color specification (default ColorSpec.UNCHANGED)")
        .def_property("chroma_subsampling", &EncodeParams::getChromaSubsampling, &EncodeParams::setChromaSubsampling,
            "Chroma subsampling (default ChromaSubsampling.CSS_444)")
        .def_property("jpeg_params", &EncodeParams::getJpegEncodeParams, &EncodeParams::setJpegEncodeParams, "Jpeg encode parameters")
        .def_property(
            "jpeg2k_params", &EncodeParams::getJpeg2kEncodeParams, &EncodeParams::setJpeg2kEncodeParams, "Jpeg2000 encode parameters");
}

} // namespace nvimgcdcs
