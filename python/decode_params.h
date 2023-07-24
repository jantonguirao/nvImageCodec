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

#include <string>
#include <vector>

#include <nvimgcodecs.h>

#include <pybind11/pybind11.h>

namespace nvimgcdcs {

namespace py = pybind11;
using namespace py::literals;

class DecodeParams
{
  public:
    DecodeParams();    
    bool getEnableOrientation() {return decode_params_.apply_exif_orientation;}
    void setEnableOrientation(bool enable){decode_params_.apply_exif_orientation = enable;};
    nvimgcdcsColorSpec_t getColorSpec() {return color_spec_;}
    void setColorSpec(nvimgcdcsColorSpec_t color_spec){color_spec_ = color_spec;};

    static void exportToPython(py::module& m);

    nvimgcdcsDecodeParams_t decode_params_;
    nvimgcdcsColorSpec_t color_spec_;
};

} // namespace nvimgcdcs
