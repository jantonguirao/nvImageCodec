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

#include <optional>
#include <string>
#include <vector>

#include <nvimgcodec.h>

#include <pybind11/pybind11.h>

#include "backend.h"
#include "decode_params.h"
#include "encode_params.h"
#include "image.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class ILogger;

class Encoder
{
  public:
    Encoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads, std::optional<std::vector<Backend>> backends,
        const std::string& options);
    Encoder(nvimgcodecInstance_t instance, ILogger* logger, int device_id, int max_num_cpu_threads,
        std::optional<std::vector<nvimgcodecBackendKind_t>> backend_kinds, const std::string& options);
    ~Encoder();

    py::object encode(py::handle image, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream);
    void encode(
        const std::string& file_name, py::handle image, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream);

    void encode(const std::vector<std::string>& file_names, const std::vector<py::handle>& images, const std::string& codec,
        std::optional<EncodeParams> params, intptr_t cuda_stream);

    py::object enter();
    void exit(const std::optional<pybind11::type>& exc_type, const std::optional<pybind11::object>& exc_value,
        const std::optional<pybind11::object>& traceback);

    static void exportToPython(py::module& m, nvimgcodecInstance_t instance, ILogger* logger);

  private:
    void convertPyImagesToImages(const std::vector<py::handle>& py_images, std::vector<Image*>* images, intptr_t cuda_stream);
    std::vector<py::bytes> encode(
        const std::vector<py::handle>& images, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream);

    void encode(const std::vector<Image*>& images, std::optional<EncodeParams> params, intptr_t cuda_stream,
        std::function<void(size_t i, nvimgcodecImageInfo_t& out_image_info, nvimgcodecCodeStream_t* code_stream)> create_code_stream,
        std::function<void(size_t i, bool skip_item, nvimgcodecCodeStream_t code_stream)> post_encode_call_back);

    std::vector<py::bytes> encode(
        const std::vector<Image*>& images, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream);
    void encode(const std::vector<std::string>& file_names, const std::vector<Image*>& images, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream);

    std::shared_ptr<std::remove_pointer<nvimgcodecEncoder_t>::type> encoder_;
    nvimgcodecInstance_t instance_;
    ILogger* logger_;
};

} // namespace nvimgcodec
