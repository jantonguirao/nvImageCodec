/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "chroma_subsampling.h"

namespace nvimgcodec {

void ChromaSubsampling::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecChromaSubsampling_t>(m, "ChromaSubsampling")
        .value("CSS_444", NVIMGCODEC_SAMPLING_444)
        .value("CSS_422", NVIMGCODEC_SAMPLING_422)
        .value("CSS_420", NVIMGCODEC_SAMPLING_420)
        .value("CSS_440", NVIMGCODEC_SAMPLING_440)
        .value("CSS_411", NVIMGCODEC_SAMPLING_411)
        .value("CSS_410", NVIMGCODEC_SAMPLING_410)
        .value("CSS_GRAY", NVIMGCODEC_SAMPLING_GRAY)
        .value("CSS_410V", NVIMGCODEC_SAMPLING_410V)
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec
