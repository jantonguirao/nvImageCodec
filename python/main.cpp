/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <nvimgcdcs_version.h>
#include <nvimgcodecs.h>
#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

class Module
{
  public:
    Module()
    {
        std::cout << "initializing nvimgcodecs" << std::endl;
        nvimgcdcsInstanceCreateInfo_t instance_create_info;
        instance_create_info.type             = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_create_info.next             = NULL;
        instance_create_info.pinned_allocator = NULL;
        instance_create_info.device_allocator = NULL;

        nvimgcdcsInstanceCreate(&instance_, instance_create_info);
    }
    ~Module()
    {
        std::cout << "finalizing nvimgcodecs" << std::endl;
        nvimgcdcsInstanceDestroy(instance_);
    }

    nvimgcdcsInstance_t instance_;
};

//TODO move to Image.hpp 
class Image
{
  public:
    Image()
        : image_(nullptr){};
    explicit Image(nvimgcdcsImage_t image)
        : image_(image)
    {
    }
    ~Image()
    {
        if (image_)
            nvimgcdcsImageDestroy(image_);
    }
    int getWidth() const
    {
        nvimgcdcsImageInfo_t image_info;
        nvimgcdcsImageGetImageInfo(image_, &image_info);
        return image_info.image_width;
    }
    int getHeight() const
    {
        nvimgcdcsImageInfo_t image_info;
        nvimgcdcsImageGetImageInfo(image_, &image_info);
        return image_info.image_height;
    }

    static Image* createImageFromFile(nvimgcdcsInstance_t instance, const char* file_name)
    {
        nvimgcdcsImage_t image;
        nvimgcdcsImgRead(instance, &image, file_name);
        return new Image(image);
    }
    nvimgcdcsImage_t image_;
};

PYBIND11_MODULE(nvimgcodecs, m)
{
    static Module module;

    m.doc() = R"pbdoc(
        nvImageCodecs Python API reference
        ========================

        This is the Python API reference for the NVIDIA® nvImageCodecs library.
    )pbdoc";

    m.attr("__version__") = "test version"; //NVIMGCDCS_VERSION_STRING;
    py::class_<Image>(m, "Image")
        .def(py::init<>())
        .def("getWidth", &Image::getWidth)
        .def("getHeight", &Image::getHeight);

    m.def("imread", [](const char* file_name) -> Image* {
        return Image::createImageFromFile(module.instance_, file_name);
    });
    m.def("imwrite", [](Image* image, const char* file_name) -> void {
        nvimgcdcsImgWrite(module.instance_, image->image_, file_name, NULL);
    });
}
