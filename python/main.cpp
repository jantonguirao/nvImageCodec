/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <nvimgcdcs_version.h>
#include <nvimgcodecs.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "module.h"
#include "image.h"
#include "decoder.h"
#include "encoder.h"


#include <iostream>

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(nvimgcodecs_impl, m)
{
    using namespace nvimgcdcs;

    static Module module;

    m.doc() = R"pbdoc(

        nvImageCodecs Python API reference

        This is the Python API reference for the NVIDIA® nvImageCodecs library.
    )pbdoc";


    nvimgcdcsProperties_t properties{NVIMGCDCS_STRUCTURE_TYPE_PROPERTIES, 0};
    nvimgcdcsGetProperties(&properties);
    std::stringstream ver_ss{};
    ver_ss << NVIMGCDCS_STREAM_VER(properties.version);
    m.attr("__version__") = ver_ss.str();
   
    Module::exportToPython(m, module.instance_);
    Image::exportToPython(m);
    Decoder::exportToPython(m, module.instance_);
    Encoder::exportToPython(m, module.instance_);
}