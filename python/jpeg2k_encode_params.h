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
#include <tuple>

#include <nvimgcodecs.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace nvimgcdcs {

namespace py = pybind11;
using namespace py::literals;

class Jpeg2kEncodeParams
{
  public:
    Jpeg2kEncodeParams();

    bool getJpeg2kReversible() { return !nvimgcdcs_jpeg2k_encode_params_.irreversible; }
    void setJpeg2kReversible(bool reversible) { nvimgcdcs_jpeg2k_encode_params_.irreversible = !reversible; }

    std::tuple<int, int> getJpeg2kCodeBlockSize(){
        return std::make_tuple<int, int>(nvimgcdcs_jpeg2k_encode_params_.code_block_w, nvimgcdcs_jpeg2k_encode_params_.code_block_h);
    }
    void setJpeg2kCodeBlockSize(std::tuple<int, int> size) {
        nvimgcdcs_jpeg2k_encode_params_.code_block_w = std::get<0>(size);
        nvimgcdcs_jpeg2k_encode_params_.code_block_h = std::get<1>(size);
    }
    int getJpeg2kNumResoulutions() { return nvimgcdcs_jpeg2k_encode_params_.num_resolutions; }
    void setJpeg2kNumResoulutions(int num_resolutions) { nvimgcdcs_jpeg2k_encode_params_.num_resolutions = num_resolutions; };

    nvimgcdcsJpeg2kBitstreamType_t getJpeg2kBitstreamType() { return nvimgcdcs_jpeg2k_encode_params_.stream_type; }
    void setJpeg2kBitstreamType(nvimgcdcsJpeg2kBitstreamType_t bistream_type) {
        nvimgcdcs_jpeg2k_encode_params_.stream_type = bistream_type;
    };

    nvimgcdcsJpeg2kProgOrder_t getJpeg2kProgOrder() { return nvimgcdcs_jpeg2k_encode_params_.prog_order; }
    void setJpeg2kProgOrder(nvimgcdcsJpeg2kProgOrder_t prog_order) { nvimgcdcs_jpeg2k_encode_params_.prog_order = prog_order; };

    static void exportToPython(py::module& m);

    nvimgcdcsJpeg2kEncodeParams_t nvimgcdcs_jpeg2k_encode_params_;
};

} // namespace nvimgcdcs
