/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "code_stream.h"

#include <iostream>

#include "error_handling.h"

namespace nvimgcodec {

CodeStream* CodeStream::CreateFromFile(nvimgcodecInstance_t instance, const char* file_name)
{
    auto ptr = std::make_unique<CodeStream>();
    auto ret = nvimgcodecCodeStreamCreateFromFile(instance, &ptr->code_stream_, file_name);
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
    return ptr.release();
}

CodeStream* CodeStream::CreateFromHostMem(nvimgcodecInstance_t instance, const unsigned char * data, size_t len)
{
    auto ptr = std::make_unique<CodeStream>();
    auto ret = nvimgcodecCodeStreamCreateFromHostMem(instance, &ptr->code_stream_, data, len);
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
    return ptr.release();
}

CodeStream* CodeStream::CreateFromHostMem(nvimgcodecInstance_t instance, py::bytes data)
{
    auto ptr = std::make_unique<CodeStream>();
    ptr->data_ref_bytes_ = data;
    auto data_view = static_cast<std::string_view>(ptr->data_ref_bytes_);
    auto ret = nvimgcodecCodeStreamCreateFromHostMem(instance, &ptr->code_stream_, reinterpret_cast<const unsigned char*>(data_view.data()), data_view.size());
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
    return ptr.release();
}

CodeStream* CodeStream::CreateFromHostMem(nvimgcodecInstance_t instance, py::array_t<uint8_t> arr)
{
    auto ptr = std::make_unique<CodeStream>();
    ptr->data_ref_arr_ = arr;
    auto data = ptr->data_ref_arr_.unchecked<1>();
    auto ret = nvimgcodecCodeStreamCreateFromHostMem(instance, &ptr->code_stream_, data.data(0), data.size());
    if (ret != NVIMGCODEC_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create code stream");
    return ptr.release();
}

CodeStream::CodeStream()
{
}

CodeStream::~CodeStream()
{
    nvimgcodecCodeStreamDestroy(code_stream_);
}

nvimgcodecCodeStream_t CodeStream::handle() const {
    return code_stream_;
}

void CodeStream::exportToPython(py::module& m, nvimgcodecInstance_t instance)
{
    py::class_<CodeStream>(m, "CodeStream")
        .def_static("CreateFromFile", [instance](const char* filename) { return CreateFromFile(instance, filename); }, R"pbdoc(
            Creates a code stream from a file path.

            Args:
                filename: Path to an image file.

            Returns:
                A code stream instance
            )pbdoc")
        .def_static("CreateFromHostMem", [instance](py::bytes data) { return CreateFromHostMem(instance, data); }, R"pbdoc(
            Creates a code stream from an encoded stream in host memory.

            Note: The user is responsible for keeping `data` available throughout the lifetime of the code stream.

            Args:
                bytes: encoded stream raw data

            Returns:
                A code stream instance
            )pbdoc")
        .def_static("CreateFromHostMem", [instance](py::array_t<uint8_t> arr) { return CreateFromHostMem(instance, arr); }, R"pbdoc(
            Creates a code stream from a numpy array containing an encoded stream raw data.

            Note: The user is responsible for keeping `data` available throughout the lifetime of the code stream.

            Args:
                array: encoded stream raw data

            Returns:
                A code stream instance
            )pbdoc");
}

} // namespace nvimgcodec
