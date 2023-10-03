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
#include <optional>

#include <nvimgcodec.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "image.h"
#include "decode_params.h"
#include "backend.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class Decoder
{
  public:
    Decoder(nvimgcodecInstance_t instance, int device_id, int max_num_cpu_threads, std::optional<std::vector<Backend>> backends,
        const std::string& options);
    Decoder(nvimgcodecInstance_t instance, int device_id, int max_num_cpu_threads,
        std::optional<std::vector<nvimgcodecBackendKind_t>> backend_kinds, const std::string& options);
    ~Decoder();

    py::object decode(const std::string& file_name, std::optional<DecodeParams> params, intptr_t cuda_stream);
    py::object decode(py::array_t<uint8_t> data, std::optional<DecodeParams> params, intptr_t cuda_stream);
    py::object decode(py::bytes data, std::optional<DecodeParams> params, intptr_t cuda_stream);

    std::vector<py::object> decode(const std::vector<std::string>& file_names, std::optional<DecodeParams> params, intptr_t cuda_stream);
    std::vector<py::object> decode(const std::vector<py::array_t<uint8_t>>& data_list, std::optional<DecodeParams> params, intptr_t cuda_stream);
    std::vector<py::object> decode(const std::vector<py::bytes>& data_list, std::optional<DecodeParams> params, intptr_t cuda_stream);

    static void exportToPython(py::module& m, nvimgcodecInstance_t instance);
  private:
    std::vector<py::object> decode(std::vector<nvimgcodecCodeStream_t>& code_streams, std::optional<DecodeParams> params, intptr_t cuda_stream);

    struct DecoderDeleter;
    std::shared_ptr<std::remove_pointer<nvimgcodecDecoder_t>::type> decoder_;
    nvimgcodecInstance_t instance_;
};

} // namespace nvimgcodec
