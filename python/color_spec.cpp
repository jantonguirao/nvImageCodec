/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "color_spec.h"

namespace nvimgcodec {

void ColorSpec::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecColorSpec_t>(m, "ColorSpec")
        .value("UNCHANGED", NVIMGCODEC_COLORSPEC_UNCHANGED)
        .value("YCC", NVIMGCODEC_COLORSPEC_SYCC)
        .value("RGB", NVIMGCODEC_COLORSPEC_SRGB)
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec
