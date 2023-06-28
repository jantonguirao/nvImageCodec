/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodecs.h>

#include <pybind11/pybind11.h>

#include "backend_kind.h"
#include "backend_params.h"

namespace nvimgcdcs {

namespace py = pybind11;
using namespace py::literals;

class Backend
{
  public:
    Backend();
    nvimgcdcsBackendKind_t getBackendKind() { return backend_.kind; }
    void setBackendKind(nvimgcdcsBackendKind_t backend_kind) { backend_.kind = backend_kind; };
    float getLoadHint() { return backend_.params.load_hint; }
    void setLoadHint(float load_hint) { backend_.params.load_hint = load_hint; };
    BackendParams getBackendParams() {
        BackendParams bp;
        bp.backend_params_ = backend_.params;
        return bp;
    }
    void setBackendParams(const BackendParams& backend_params) { backend_.params = backend_params.backend_params_; };

    static void exportToPython(py::module& m);

    nvimgcdcsBackend_t backend_;
};

} // namespace nvimgcdcs
