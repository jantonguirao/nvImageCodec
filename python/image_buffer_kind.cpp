/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "image_buffer_kind.h"

namespace nvimgcodec {

void ImageBufferKind::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecImageBufferKind_t>(m, "ImageBufferKind", "Defines buffer kind in which image data is stored.")
        .value("STRIDED_DEVICE", NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE, "GPU-accessible with planes in pitch-linear layout.") 
        .value("STRIDED_HOST", NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST, "Host-accessible with planes in pitch-linear layout.")
        .export_values();
    // clang-format on
};

} // namespace nvimgcodec
