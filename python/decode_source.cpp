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

#include "decode_source.h"
#include <iostream>
#include "error_handling.h"

namespace nvimgcodec {

DecodeSource::DecodeSource(const CodeStream* code_stream, std::optional<Region> region)
    : code_stream_(code_stream)
    , region_(region)
{
}

DecodeSource::~DecodeSource()
{
}

const CodeStream* DecodeSource::code_stream() const
{
    return code_stream_;
}
std::optional<Region> DecodeSource::region() const
{
    return region_;
}

void DecodeSource::exportToPython(py::module& m)
{
    py::class_<DecodeSource>(m, "DecodeSource")
        .def(py::init([](const CodeStream* code_stream, std::optional<Region> region) {
            return DecodeSource{code_stream, region};
        }),
            "code_stream"_a, "region"_a = py::none(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def_property_readonly("code_stream", &DecodeSource::code_stream)
        .def_property_readonly("region", &DecodeSource::region);
}

} // namespace nvimgcodec
