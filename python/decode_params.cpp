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
    : decode_params_{NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, nullptr, true, false, true}
{
}

void DecodeParams::exportToPython(py::module& m)
{
    py::class_<DecodeParams>(m, "DecodeParams")
        .def(py::init([]() { return DecodeParams{}; }), "Default constructor")
        .def(py::init([](bool apply_exif_orientation, bool enable_color_conversion) {
            DecodeParams p;
            p.decode_params_.apply_exif_orientation = apply_exif_orientation;
            p.decode_params_.enable_color_conversion = enable_color_conversion;
            return p;
        }),
            "apply_exif_orientation"_a = true, "enable_color_conversion"_a = true,
            "Constructor with apply_exif_orientation and enable_color_conversion parameters")
        .def_property("apply_exif_orientation", &DecodeParams::getEnableOrientation, &DecodeParams::setEnableOrientation,
            "Apply EXIF orientation if available")
        .def_property("enable_color_conversion", &DecodeParams::getEnableColorConversion, &DecodeParams::setEnableColorConversion,
            "Enable color conversion to RGB");
}

} // namespace nvimgcdcs
