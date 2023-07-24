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

namespace nvimgcdcs {

namespace py = pybind11;
using namespace py::literals;

class EncodeParams
{
  public:
    EncodeParams();
    EncodeParams(const EncodeParams& that) { operator=(that); }

    EncodeParams& operator=(const EncodeParams& that)
    {
        if (this == &that) {
            return *this;
        }

        jpeg2k_encode_params_ = that.jpeg2k_encode_params_;
        jpeg_encode_params_ = that.jpeg_encode_params_;
        encode_params_ = that.encode_params_;
        jpeg_image_info_ = that.jpeg_image_info_;
        chroma_subsampling_ = that.chroma_subsampling_;

        jpeg2k_encode_params_.next = nullptr;
        jpeg_encode_params_.next = &jpeg2k_encode_params_;
        encode_params_.next = &jpeg_encode_params_;
        jpeg_image_info_.next = nullptr;
        return *this;
    }

    float getQuality() { return encode_params_.quality; }
    void setQuality(float quality) { encode_params_.quality = quality; };

    float getTargetPsnr() { return encode_params_.target_psnr; }
    void setTargetPsnr(float target_psnr) { encode_params_.target_psnr = target_psnr; };

    nvimgcdcsColorSpec_t getColorSpec() { return color_spec_; }
    void setColorSpec(nvimgcdcsColorSpec_t color_spec) { color_spec_ = color_spec; };

    nvimgcdcsChromaSubsampling_t getChromaSubsampling() { return chroma_subsampling_; }
    void setChromaSubsampling(nvimgcdcsChromaSubsampling_t chroma_subsampling) { chroma_subsampling_ = chroma_subsampling; }

    bool getJpegProgressive(){
        return jpeg_image_info_.encoding == NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;}
    void setJpegProgressive(bool progressive) {
        jpeg_image_info_.encoding =
            progressive ? NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN : NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT; }

    bool getJpegOptimizedHuffman(){
        return jpeg_encode_params_.optimized_huffman;}
    void setJpegOptimizedHuffman(bool optimized_huffman) {
       jpeg_encode_params_.optimized_huffman = optimized_huffman; }

    bool getJpeg2kReversible(){
        return !jpeg2k_encode_params_.irreversible;}
    void setJpeg2kReversible(bool reversible) {
        jpeg2k_encode_params_.irreversible = !reversible; }

    std::tuple<int, int> getJpeg2kCodeBlockSize(){
        return std::make_tuple<int, int>(jpeg2k_encode_params_.code_block_w, jpeg2k_encode_params_.code_block_h);
    }
    void setJpeg2kCodeBlockSize(std::tuple<int, int> size) {
        jpeg2k_encode_params_.code_block_w = std::get<0>(size);
        jpeg2k_encode_params_.code_block_h = std::get<1>(size);
    }
    int getJpeg2kNumResoulutions() { return jpeg2k_encode_params_.num_resolutions; }
    void setJpeg2kNumResoulutions(int num_resolutions) { jpeg2k_encode_params_.num_resolutions = num_resolutions; };

    nvimgcdcsJpeg2kBitstreamType_t getJpeg2kBitstreamType() { return jpeg2k_encode_params_.stream_type; }
    void setJpeg2kBitstreamType(nvimgcdcsJpeg2kBitstreamType_t bistream_type) { jpeg2k_encode_params_.stream_type = bistream_type; };

    nvimgcdcsJpeg2kProgOrder_t getJpeg2kProgOrder() { return jpeg2k_encode_params_.prog_order; }
    void setJpeg2kProgOrder(nvimgcdcsJpeg2kProgOrder_t prog_order) { jpeg2k_encode_params_.prog_order = prog_order; };

    static void exportToPython(py::module& m);

    nvimgcdcsJpeg2kEncodeParams_t jpeg2k_encode_params_;
    nvimgcdcsJpegEncodeParams_t jpeg_encode_params_;
    nvimgcdcsEncodeParams_t encode_params_;
    nvimgcdcsJpegImageInfo_t jpeg_image_info_;
    nvimgcdcsChromaSubsampling_t chroma_subsampling_;
    nvimgcdcsColorSpec_t color_spec_;
};

} // namespace nvimgcdcs
