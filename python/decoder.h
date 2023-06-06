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

    std::vector<Image> decode(const std::vector<std::string>& data_list);

    static void exportToPython(py::module& m, nvimgcdcsInstance_t instance);

  private:
    struct DecoderDeleter;
    std::shared_ptr<std::remove_pointer<nvimgcdcsDecoder_t>::type> decoder_;
    nvimgcdcsInstance_t instance_;
};

} // namespace nvimgcdcs
