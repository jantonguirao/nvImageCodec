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
    : decode_params_{NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, nullptr, true, false, true, 0, nullptr}
{
}

void DecodeParams::exportToPython(py::module& m)
{
    py::class_<DecodeParams>(m, "DecodeParams")
        .def(py::init([]() { return DecodeParams{}; }), "Default constructor")
        .def(py::init([](bool enable_orientation) {
            DecodeParams p;
            p.decode_params_.enable_orientation = enable_orientation;
            return p;
        }),
            "enable_orientation"_a, "Constructor with enable_orientation parameter")
        .def_property("enable_orientation", &DecodeParams::getEnableOrientation, &DecodeParams::setEnableOrientation,
            "Apply EXIF orientation if available");
}

} // namespace nvimgcdcs
