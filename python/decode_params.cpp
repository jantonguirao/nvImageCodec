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
{
}

void DecodeParams::exportToPython(py::module& m)
{
    py::class_<DecodeParams>(m, "DecodeParams")
        .def(py::init([]() { return DecodeParams{}; }), "Default constructor")
        .def(py::init([](bool apply_exif_orientation, nvimgcdcsColorSpec_t color_spec) {
            DecodeParams p;
            p.decode_params_.apply_exif_orientation = apply_exif_orientation;
            p.color_spec_ = color_spec;
            return p;
        }),
            "apply_exif_orientation"_a = true, "color_spec"_a = NVIMGCDCS_COLORSPEC_SRGB,
            "Constructor with apply_exif_orientation and color_spec parameters")
        .def_property("apply_exif_orientation", &DecodeParams::getEnableOrientation, &DecodeParams::setEnableOrientation,
            "Apply EXIF orientation if available")
        .def_property("color_spec", &DecodeParams::getColorSpec, &DecodeParams::setColorSpec,
            "Color specification");
}

} // namespace nvimgcdcs
