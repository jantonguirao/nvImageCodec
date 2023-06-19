/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "mct_mode.h"

namespace nvimgcdcs {

void MctMode::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcdcsMctMode_t>(m, "MctMode")
        .value("YCC", NVIMGCDCS_MCT_MODE_YCC)
        .value("RGB", NVIMGCDCS_MCT_MODE_RGB )
        .export_values();
    // clang-format on
}

} // namespace nvimgcdcs
