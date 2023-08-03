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

#include <nvimgcodecs.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "image.h"
#include "decode_params.h"
#include "backend.h"

namespace nvimgcdcs {

namespace py = pybind11;
using namespace py::literals;

class Decoder
{
  public:
    Decoder(nvimgcdcsInstance_t instance, int device_id, int num_cpu_threads, std::optional<std::vector<Backend>> backends,
        const std::string& options);
    Decoder(nvimgcdcsInstance_t instance, int device_id, int num_cpu_threads,
        std::optional<std::vector<nvimgcdcsBackendKind_t>> backend_kinds, const std::string& options);
    ~Decoder();

    Image decode(const std::string& file_name, std::optional<DecodeParams> params, intptr_t cuda_stream);
    Image decode(py::array_t<uint8_t> data, std::optional<DecodeParams> params, intptr_t cuda_stream);
    Image decode(py::bytes data, std::optional<DecodeParams> params, intptr_t cuda_stream);
    std::vector<Image> decode(const std::vector<std::string>& file_names, std::optional<DecodeParams> params, intptr_t cuda_stream);
    std::vector<Image> decode(const std::vector<py::array_t<uint8_t>>& data_list, std::optional<DecodeParams> params, intptr_t cuda_stream);
    std::vector<Image> decode(const std::vector<py::bytes>& data_list, std::optional<DecodeParams> params, intptr_t cuda_stream);
    std::vector<Image> decode(std::vector<nvimgcdcsCodeStream_t>& code_streams, std::optional<DecodeParams> params, intptr_t cuda_stream);
    static void exportToPython(py::module& m, nvimgcdcsInstance_t instance);

  private:
    struct DecoderDeleter;
    std::shared_ptr<std::remove_pointer<nvimgcdcsDecoder_t>::type> decoder_;
    nvimgcdcsInstance_t instance_;
};

} // namespace nvimgcdcs
