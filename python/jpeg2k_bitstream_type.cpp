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

namespace nvimgcodec {

void Jpeg2kBitstreamType::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecJpeg2kBitstreamType_t>(m, "Jpeg2kBitstreamType")
        .value("J2K", NVIMGCODEC_JPEG2K_STREAM_J2K)
        .value("JP2", NVIMGCODEC_JPEG2K_STREAM_JP2)
        .export_values();
    // clang-format on
};

} // namespace nvimgcodec
