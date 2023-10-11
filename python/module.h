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

#include <nvimgcodec.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <default_debug_messenger.h>
#include <logger.h>

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;


class Module
{
  public:
    Module();
    ~Module();

    static void exportToPython(py::module& m, nvimgcodecInstance_t instance);

    nvimgcodecInstance_t instance_;
    std::unique_ptr<DefaultDebugMessenger> dbg_messenger_;
    std::unique_ptr<Logger> logger_;
    nvimgcodecDebugMessenger_t dbg_messenger_handle_;
};

} // namespace nvimgcodec
