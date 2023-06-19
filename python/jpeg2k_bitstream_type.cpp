/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "jpeg2k_bitstream_type.h"

namespace nvimgcdcs {

void Jpeg2kBitstreamType::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcdcsJpeg2kBitstreamType_t>(m, "Jpeg2kBitstreamType")
        .value("J2K", NVIMGCDCS_JPEG2K_STREAM_J2K)
        .value("JP2", NVIMGCDCS_JPEG2K_STREAM_JP2)
        .export_values();
    // clang-format on
};

} // namespace nvimgcdcs
