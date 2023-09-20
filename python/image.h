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

#include "dlpack_utils.h"

namespace nvimgcdcs {

namespace py = pybind11;
using namespace py::literals;

class Image
{
  public:
    Image(nvimgcdcsInstance_t instance, nvimgcdcsImageInfo_t* image_info);
    Image(nvimgcdcsInstance_t instance, PyObject* o, intptr_t cuda_stream);

    int getWidth() const;
    int getHeight() const;
    int getNdim() const;
    nvimgcdcsImageBufferKind_t getBufferKind() const;

    py::dict array_interface() const;
    py::dict cuda_interface() const;

    py::object shape() const;
    py::object dtype() const;

    py::capsule dlpack(py::object stream) const;
    const py::tuple getDlpackDevice() const;

    py::object cpu();
    py::object cuda();

    nvimgcdcsImage_t getNvImgCdcsImage() const;
    static void exportToPython(py::module& m);

  private:
    void initImageInfoFromInterfaceDict(const py::dict& d, nvimgcdcsImageInfo_t* image_info);
    void initInterfaceDictFromImageInfo(const nvimgcdcsImageInfo_t& image_info, py::dict* d);
    void initArrayInterface(const nvimgcdcsImageInfo_t& image_info);
    void initCudaArrayInterface(const nvimgcdcsImageInfo_t& image_info);
    void initCudaEventForDLPack();
    void initDLPack(nvimgcdcsImageInfo_t* image_info, py::capsule cap);
    void initBuffer(nvimgcdcsImageInfo_t* image_info);
    void initDeviceBuffer(nvimgcdcsImageInfo_t* image_info);
    void initHostBuffer(nvimgcdcsImageInfo_t* image_info);

    nvimgcdcsInstance_t instance_;
    std::vector<unsigned char> img_host_buffer_;
    std::shared_ptr<unsigned char> img_buffer_;
    std::shared_ptr<std::remove_pointer<nvimgcdcsImage_t>::type> image_;
    py::dict array_interface_;
    py::dict cuda_array_interface_;
    std::shared_ptr<DLPackTensor> dlpack_tensor_;
    std::shared_ptr<std::remove_pointer<cudaEvent_t>::type> dlpack_cuda_event_;
};

} // namespace nvimgcdcs
