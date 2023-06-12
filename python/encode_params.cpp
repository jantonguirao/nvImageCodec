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

EncodeParams::EncodeParams()
    : jpeg2k_encode_params_{NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, 0, NVIMGCDCS_JPEG2K_STREAM_JP2, NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL,
          5, 64, 64, true}
    , jpeg_encode_params_{NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, &jpeg2k_encode_params_, false}
    , encode_params_{NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS, &jpeg_encode_params_, 95, 50, NVIMGCDCS_MCT_MODE_RGB, 0, nullptr}
    , jpeg_image_info_{NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0, NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT}
    , chroma_subsampling_{NVIMGCDCS_SAMPLING_444}
{
    jpeg_encode_params_.next = &jpeg2k_encode_params_;
    encode_params_.next = &jpeg_encode_params_;
    jpeg_image_info_.next = 0;
}

void EncodeParams::exportToPython(py::module& m)
{
    py::class_<EncodeParams>(m, "EncodeParams")
        .def(py::init([]() { return EncodeParams{}; }), "Default constructor")
        .def(py::init([](float quality, float target_psnr, nvimgcdcsMctMode_t mct_mode, nvimgcdcsChromaSubsampling_t chroma_subsampling,
                          bool jpeg_progressive, bool jpeg_optimized_huffman, bool jpeg2k_reversible,
                          std::tuple<int, int> jpeg2k_code_block_size, int jpeg2k_num_resolutions,
                          nvimgcdcsJpeg2kBitstreamType_t jpeg2k_bitstream_type, nvimgcdcsJpeg2kProgOrder_t jpeg2k_prog_order) {
            EncodeParams p;
            p.encode_params_.quality = quality;
            p.encode_params_.target_psnr = target_psnr;
            p.encode_params_.mct_mode = mct_mode;
            p.chroma_subsampling_ = chroma_subsampling;
            p.jpeg_encode_params_.optimized_huffman = jpeg_optimized_huffman;
            p.jpeg_image_info_.encoding =
                jpeg_progressive ? NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN : NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT;
            p.jpeg2k_encode_params_.irreversible = !jpeg2k_reversible;
            p.jpeg2k_encode_params_.code_block_w = std::get<0>(jpeg2k_code_block_size);
            p.jpeg2k_encode_params_.code_block_h = std::get<1>(jpeg2k_code_block_size);
            p.jpeg2k_encode_params_.num_resolutions = jpeg2k_num_resolutions;
            p.jpeg2k_encode_params_.stream_type = jpeg2k_bitstream_type;
            p.jpeg2k_encode_params_.prog_order = jpeg2k_prog_order;

            return p;
        }),
            // clang-format off
            "quality"_a = 95, 
            "target_psnr"_a = 50, 
            "mct_mode"_a = NVIMGCDCS_MCT_MODE_RGB, 
            "chroma_subsampling"_a = NVIMGCDCS_SAMPLING_444,
            "jpeg_progressive"_a = false,
            "jpeg_optimized_huffman"_a = false,
            "jpeg2k_reversible"_a = false,
            "jpeg2k_code_block_size"_a = std::make_tuple<int, int>(64, 64), 
            "jpeg2k_num_resolutions"_a = 5,
            "jpeg2k_bitstream_type"_a = NVIMGCDCS_JPEG2K_STREAM_JP2, 
            "jpeg2k_prog_order"_a = NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL,
            "Constructor with quality, target_psnr, mct_mode, chroma_subsampling etc. parameters")
        // clang-format on
        .def_property("quality", &EncodeParams::getQuality, &EncodeParams::setQuality, "Quality value 0-100 (default 95)")
        .def_property("target_psnr", &EncodeParams::getTargetPsnr, &EncodeParams::setTargetPsnr, "Target psnr (default 50)")
        .def_property(
            "mct_mode", &EncodeParams::getMctMode, &EncodeParams::setMctMode, "Multi-Color Transform mode value (default MctMode.RGB)")
        .def_property("chroma_subsampling", &EncodeParams::getChromaSubsampling, &EncodeParams::setChromaSubsampling,
            "Chroma subsampling (default ChromaSubsampling.CSS_444)")
        .def_property("jpeg_progressive", &EncodeParams::getJpegProgressive, &EncodeParams::setJpegProgressive,
            "Use Jpeg progressive encoding (default False)")
        .def_property("jpeg_optimized_huffman", &EncodeParams::getJpegOptimizedHuffman, &EncodeParams::setJpegOptimizedHuffman,
            "Use Jpeg encoding with optimized huffman (default False)")
        .def_property("jpeg2k_reversible", &EncodeParams::getJpeg2kReversible, &EncodeParams::setJpeg2kReversible,
            "Use reversible Jpeg 2000 transform (default False)")
        .def_property("jpeg2k_code_block_size", &EncodeParams::getJpeg2kCodeBlockSize, &EncodeParams::setJpeg2kCodeBlockSize,
            "Jpeg 2000 code block width and height (default 64x64)")
        .def_property("jpeg2k_num_resolutions", &EncodeParams::getJpeg2kNumResoulutions, &EncodeParams::setJpeg2kNumResoulutions,
            "Jpeg 2000 number of resoultions (decomposition levels) (default 5)")
        .def_property("jpeg2k_bitstream_type", &EncodeParams::getJpeg2kBitstreamType, &EncodeParams::setJpeg2kBitstreamType,
            "Jpeg 2000 bitstream type (default JP2)")
        .def_property("jpeg2k_prog_order", &EncodeParams::getJpeg2kProgOrder, &EncodeParams::setJpeg2kProgOrder,
            "Jpeg 2000 progression order (default RPCL)");
}

} // namespace nvimgcdcs
