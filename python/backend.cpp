/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "backend.h"
#include <iostream>

#include "error_handling.h"

namespace nvimgcodec {

Backend::Backend()
    : backend_{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, nullptr, NVIMGCODEC_BACKEND_KIND_GPU_ONLY,
          {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, nullptr, 1.0f}}
{
}

void Backend::exportToPython(py::module& m)
{
    py::class_<Backend>(m, "Backend")
        .def(py::init([]() { return Backend{}; }), "Default constructor")
        .def(py::init([](nvimgcodecBackendKind_t backend_kind, float load_hint) {
            Backend p;
            p.backend_.kind = backend_kind;
            p.backend_.params.load_hint = load_hint;
            return p;
        }),
            "backend_kind"_a, "load_hint"_a = 1.0f, "Constructor with parameters")
        .def(py::init([](nvimgcodecBackendKind_t backend_kind, BackendParams backend_params) {
            Backend p;
            p.backend_.kind = backend_kind;
            p.backend_.params = backend_params.backend_params_;
            return p;
        }),
            "backend_kind"_a, "backend_params"_a, "Constructor with backend parameters")
        .def_property("backend_kind", &Backend::getBackendKind, &Backend::setBackendKind, "Backend kind (e.g. GPU_ONLY or CPU_ONLY).")
        .def_property("load_hint", &Backend::getLoadHint, &Backend::setLoadHint,
            "Fraction of the batch samples that will be picked by this backend. The remaining samples will be picked by the next lower "
            "priority backend.")
        .def_property("backend_params", &Backend::getBackendParams, &Backend::setBackendParams, "Backend parameters.");
}
 
} // namespace nvimgcodec
