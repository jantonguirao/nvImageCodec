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

#pragma once

#include <nvimgcodec.h>

#include <pybind11/pybind11.h>

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class Region
{
  public:
    static void exportToPython(py::module& m);

    int ndim() const {
      return region.ndim;
    }

    py::tuple start() const {
      auto ret = py::tuple(ndim());
      for (int i = 0; i < ndim(); i++) {
        ret[i] = region.start[i];
      }
      return ret;
    }

    py::tuple end() const {
      auto ret = py::tuple(ndim());
      for (int i = 0; i < ndim(); i++) {
        ret[i] = region.end[i];
      }
      return ret;
    }

    nvimgcodecRegion_t region = {NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 0};
};

} // namespace nvimgcodec
