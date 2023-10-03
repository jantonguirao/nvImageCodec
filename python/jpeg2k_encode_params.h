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

#include <nvimgcodec.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class Jpeg2kEncodeParams
{
  public:
    Jpeg2kEncodeParams();

    bool getJpeg2kReversible() { return !nvimgcodec_jpeg2k_encode_params_.irreversible; }
    void setJpeg2kReversible(bool reversible) { nvimgcodec_jpeg2k_encode_params_.irreversible = !reversible; }

    std::tuple<int, int> getJpeg2kCodeBlockSize(){
        return std::make_tuple<int, int>(nvimgcodec_jpeg2k_encode_params_.code_block_w, nvimgcodec_jpeg2k_encode_params_.code_block_h);
    }
    void setJpeg2kCodeBlockSize(std::tuple<int, int> size) {
        nvimgcodec_jpeg2k_encode_params_.code_block_w = std::get<0>(size);
        nvimgcodec_jpeg2k_encode_params_.code_block_h = std::get<1>(size);
    }
    int getJpeg2kNumResoulutions() { return nvimgcodec_jpeg2k_encode_params_.num_resolutions; }
    void setJpeg2kNumResoulutions(int num_resolutions) { nvimgcodec_jpeg2k_encode_params_.num_resolutions = num_resolutions; };

    nvimgcodecJpeg2kBitstreamType_t getJpeg2kBitstreamType() { return nvimgcodec_jpeg2k_encode_params_.stream_type; }
    void setJpeg2kBitstreamType(nvimgcodecJpeg2kBitstreamType_t bistream_type) {
        nvimgcodec_jpeg2k_encode_params_.stream_type = bistream_type;
    };

    nvimgcodecJpeg2kProgOrder_t getJpeg2kProgOrder() { return nvimgcodec_jpeg2k_encode_params_.prog_order; }
    void setJpeg2kProgOrder(nvimgcodecJpeg2kProgOrder_t prog_order) { nvimgcodec_jpeg2k_encode_params_.prog_order = prog_order; };

    static void exportToPython(py::module& m);

    nvimgcodecJpeg2kEncodeParams_t nvimgcodec_jpeg2k_encode_params_;
};

} // namespace nvimgcodec
