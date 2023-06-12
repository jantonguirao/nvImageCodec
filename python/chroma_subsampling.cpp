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

namespace nvimgcdcs {

void ChromaSubsampling::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcdcsChromaSubsampling_t>(m, "ChromaSubsampling")
        .value("CSS_444", NVIMGCDCS_SAMPLING_444)
        .value("CSS_422", NVIMGCDCS_SAMPLING_422)
        .value("CSS_420", NVIMGCDCS_SAMPLING_420)
        .value("CSS_440", NVIMGCDCS_SAMPLING_440)
        .value("CSS_411", NVIMGCDCS_SAMPLING_411)
        .value("CSS_410", NVIMGCDCS_SAMPLING_410)
        .value("CSS_GRAY", NVIMGCDCS_SAMPLING_GRAY)
        .value("CSS_410V", NVIMGCDCS_SAMPLING_410V)
        .export_values();
    // clang-format on
}

} // namespace nvimgcdcs
