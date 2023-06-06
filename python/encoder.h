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

class Encoder
{
  public:
    explicit Encoder(nvimgcdcsInstance_t instance, int device_id, const std::string& options);
    ~Encoder();

    std::vector<py::bytes> encode(const std::string& ext, const std::vector<Image>& images);

    static void exportToPython(py::module& m, nvimgcdcsInstance_t instance);

  private:
    struct EncoderDeleter;
    std::shared_ptr<std::remove_pointer<nvimgcdcsEncoder_t>::type> encoder_;
    nvimgcdcsInstance_t instance_;
};

} // namespace nvimgcdcs
