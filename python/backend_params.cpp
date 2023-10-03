/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "backend_params.h"

#include <iostream>

#include "error_handling.h"

namespace nvimgcodec {

BackendParams::BackendParams()
    : backend_params_{NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, nullptr, 1.0f}
{
}

void BackendParams::exportToPython(py::module& m)
{
    py::class_<BackendParams>(m, "BackendParams")
        .def(py::init([]() { return BackendParams{}; }), "Default constructor")
        .def(py::init([](bool load_hint) {
            BackendParams p;
            p.backend_params_.load_hint = load_hint;
            return p;
        }),
            "load_hint"_a = 1.0f, "Constructor with load_hint parameters")
        .def_property("load_hint", &BackendParams::getLoadHint, &BackendParams::setLoadHint,
            "Fraction of the batch samples that will be picked by this backend. The remaining samples will be picked by the next lower "
            "priority backend. This is just hint so particular codec can ignore this "
            "value");
}

} // namespace nvimgcodec
