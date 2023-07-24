/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "encode_params.h"

#include <iostream>

#include "error_handling.h"

namespace nvimgcdcs {

JpegEncodeParams::JpegEncodeParams()
    : nvimgcdcs_jpeg_encode_params_{NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, nullptr, false}
    , nvimgcdcs_jpeg_image_info_{NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, nullptr, NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT}
{
}

void JpegEncodeParams::exportToPython(py::module& m)
{
    py::class_<JpegEncodeParams>(m, "JpegEncodeParams")
        .def(py::init([]() { return JpegEncodeParams{}; }), "Default constructor")
        .def(py::init([](bool jpeg_progressive, bool jpeg_optimized_huffman) {
            JpegEncodeParams p;
            p.nvimgcdcs_jpeg_encode_params_.optimized_huffman = jpeg_optimized_huffman;
            p.nvimgcdcs_jpeg_image_info_.encoding =
                jpeg_progressive ? NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN : NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT;
            return p;
        }),
            // clang-format off
            "progressive"_a = false,
            "optimized_huffman"_a = false,
            "Constructor with progressive, optimized_huffman parameters")
        // clang-format on
        .def_property("progressive", &JpegEncodeParams::getJpegProgressive, &JpegEncodeParams::setJpegProgressive,
            "Use Jpeg progressive encoding (default False)")
        .def_property("optimized_huffman", &JpegEncodeParams::getJpegOptimizedHuffman, &JpegEncodeParams::setJpegOptimizedHuffman,
            "Use Jpeg encoding with optimized Huffman (default False)");
}

} // namespace nvimgcdcs
