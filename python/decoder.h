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

namespace nvimgcdcs {

namespace py = pybind11;

class Decoder
{
  public:
    explicit Decoder(nvimgcdcsInstance_t instance, int device_id, const std::string& options);
    ~Decoder();

    Image decode(const std::string& file_name, int cuda_stream);
    Image decode(py::bytes data, int cuda_stream);
    std::vector<Image> decode(const std::vector<std::string>& file_names, int cuda_stream);
    std::vector<Image> decode(const std::vector<py::bytes>& data_list, int cuda_stream);
    std::vector<Image> decode(std::vector<nvimgcdcsCodeStream_t>& code_streams, int cuda_stream);
    static void exportToPython(py::module& m, nvimgcdcsInstance_t instance);

  private:
    struct DecoderDeleter;
    std::shared_ptr<std::remove_pointer<nvimgcdcsDecoder_t>::type> decoder_;
    nvimgcdcsInstance_t instance_;
};

} // namespace nvimgcdcs
