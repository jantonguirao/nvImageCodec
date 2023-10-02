/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "decode_params.h"

#include <iostream>

#include "error_handling.h"

namespace nvimgcdcs {

DecodeParams::DecodeParams()
    : decode_params_{NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, nullptr, true, false}
    , color_spec_{NVIMGCDCS_COLORSPEC_SRGB}
    , allow_any_depth_{false}
{
}

void DecodeParams::exportToPython(py::module& m)
{
    py::class_<DecodeParams>(m, "DecodeParams")
        .def(py::init([]() { return DecodeParams{}; }), "Default constructor")
        .def(py::init([](bool apply_exif_orientation, nvimgcdcsColorSpec_t color_spec, bool allow_any_depth) {
            DecodeParams p;
            p.decode_params_.apply_exif_orientation = apply_exif_orientation;
            p.color_spec_ = color_spec;
            p.allow_any_depth_ = allow_any_depth;
            return p;
        }),
            "apply_exif_orientation"_a = true, "color_spec"_a = NVIMGCDCS_COLORSPEC_SRGB, "allow_any_depth"_a = false,
            "Constructor with apply_exif_orientation, color_spec parameters, and allow_any_depth")
        .def_property("apply_exif_orientation", &DecodeParams::getEnableOrientation, &DecodeParams::setEnableOrientation,
            "Apply EXIF orientation if available")
        .def_property("allow_any_depth", &DecodeParams::getAllowAnyDepth, &DecodeParams::setAllowAnyDepth,
            "Allow any native bitdepth. If not enabled, the dynamic range is scaled to uint8.")
        .def_property("color_spec", &DecodeParams::getColorSpec, &DecodeParams::setColorSpec,
            "Color specification");
}

} // namespace nvimgcdcs
