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

class JpegEncodeParams
{
  public:
    JpegEncodeParams();

    bool getJpegProgressive(){
        return nvimgcodec_jpeg_image_info_.encoding == NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;}
    void setJpegProgressive(bool progressive) {
        nvimgcodec_jpeg_image_info_.encoding =
            progressive ? NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN : NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT; }

    bool getJpegOptimizedHuffman(){
        return nvimgcodec_jpeg_encode_params_.optimized_huffman;}
    void setJpegOptimizedHuffman(bool optimized_huffman) {
       nvimgcodec_jpeg_encode_params_.optimized_huffman = optimized_huffman; }

    static void exportToPython(py::module& m);

    nvimgcodecJpegEncodeParams_t nvimgcodec_jpeg_encode_params_;
    nvimgcodecJpegImageInfo_t nvimgcodec_jpeg_image_info_;
};

} // namespace nvimgcodec
