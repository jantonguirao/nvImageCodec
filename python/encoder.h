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

#include "image.h"
#include "encode_params.h"
#include "decode_params.h"
#include "backend.h"

namespace nvimgcdcs {

namespace py = pybind11;
using namespace py::literals;

class Encoder
{
  public:
    Encoder(nvimgcdcsInstance_t instance, int device_id, const std::vector<Backend>& backends, const std::string& options);
    Encoder(nvimgcdcsInstance_t instance, int device_id, const std::vector<nvimgcdcsBackendKind_t>& backend_kinds, const std::string& options);
    ~Encoder();

    py::bytes encode(Image image, const std::string& codec, const EncodeParams& params, intptr_t cuda_stream);
    void encode(const std::string& file_name, Image image,  const std::string& codec, const EncodeParams& params, intptr_t cuda_stream);
    std::vector<py::bytes> encode(const std::vector<Image>& images, const std::string& codec, const EncodeParams& params, intptr_t cuda_stream);
    void encode(const std::vector<std::string>& file_names, const std::vector<Image>& images, const std::string& codec, const EncodeParams& params, intptr_t cuda_stream);

    static void exportToPython(py::module& m, nvimgcdcsInstance_t instance);

  private:
    void encode(const std::vector<Image>& images, const EncodeParams& params, intptr_t cuda_stream,
        std::function<void(size_t i, nvimgcdcsImageInfo_t& out_image_info, nvimgcdcsCodeStream_t* code_stream)> create_code_stream,
        std::function<void(size_t i, bool skip_item, nvimgcdcsCodeStream_t code_stream)> post_encode_call_back);
    struct EncoderDeleter;
    std::shared_ptr<std::remove_pointer<nvimgcdcsEncoder_t>::type> encoder_;
    nvimgcdcsInstance_t instance_;
};

} // namespace nvimgcdcs
