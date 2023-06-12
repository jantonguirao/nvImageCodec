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
#include <memory>

#include <nvimgcodecs.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace nvimgcdcs {

namespace py = pybind11;
using namespace py::literals;

class Image
{
  public:
    explicit Image(nvimgcdcsImage_t image);
    Image(nvimgcdcsInstance_t instance, nvimgcdcsImageInfo_t* image_info);
    Image(nvimgcdcsInstance_t instance, PyObject* o);

    int getWidth() const;
    int getHeight() const;
    int getNdim() const;

    py::dict cuda_interface() const;
    py::object shape() const;
    py::object dtype() const;

    nvimgcdcsImage_t getNvImgCdcsImage() const;
    static void exportToPython(py::module& m);

  private:
    struct BufferDeleter;
    struct ImageDeleter;
    void initCudaArrayInterface(nvimgcdcsImageInfo_t* image_info);

    size_t img_buffer_size_;
    std::shared_ptr<unsigned char> img_buffer_;
    std::shared_ptr<std::remove_pointer<nvimgcdcsImage_t>::type> image_;
    py::dict cuda_array_interface_;
};

} // namespace nvimgcdcs
