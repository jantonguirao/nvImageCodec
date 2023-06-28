/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "backend_kind.h"

namespace nvimgcdcs {

void BackendKind::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcdcsBackendKind_t>(m, "BackendKind")
        .value("CPU_ONLY", NVIMGCDCS_BACKEND_KIND_CPU_ONLY)
        .value("GPU_ONLY", NVIMGCDCS_BACKEND_KIND_GPU_ONLY)
        .value("HYBRID_CPU_GPU", NVIMGCDCS_BACKEND_KIND_HYBRID_CPU_GPU)
        .value("HW_GPU_ONLY", NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY)
        .export_values();
    // clang-format on
};

} // namespace nvimgcdcs
