/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <nvimgcdcs_version.h>
#include <nvimgcodecs.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>

namespace py = pybind11;
using namespace py::literals;

static void check_cuda_buffer(const void* ptr)
{
    if (ptr == nullptr) {
        throw std::runtime_error("NULL CUDA buffer not accepted");
    }

    cudaPointerAttributes attrs = {};
    cudaError_t err             = cudaPointerGetAttributes(&attrs, ptr);
    cudaGetLastError(); // reset the cuda error (if any)
    if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered) {
        throw std::runtime_error("Buffer is not CUDA-accessible");
    }
}

class Module
{
  public:
    Module()
    {
        nvimgcdcsInstanceCreateInfo_t instance_create_info;
        instance_create_info.type             = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_create_info.next             = NULL;
        instance_create_info.pinned_allocator = NULL;
        instance_create_info.device_allocator = NULL;

        nvimgcdcsInstanceCreate(&instance_, instance_create_info);
    }
    ~Module()
    {
        nvimgcdcsInstanceDestroy(instance_);
    }

    nvimgcdcsInstance_t instance_;
};

static std::string format_str_from_type(nvimgcdcsSampleDataType_t type)
{
    switch (type) {
    case NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8:
        return "=b";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8:
        return "=B";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16:
        return "=h";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16:
        return "=H";
        /*  case NVIMGCDCS_INT32:
        return "=i";
    case NVIMGCDCS_UINT32:
        return "=I";
    case NVIMGCDCS_INT64:
        return "=q";
    case NVIMGCDCS_UINT64:
        return "=Q";
    case NVIMGCDCS_FLOAT:
        return "=f";
    case NVIMGCDCS_FLOAT16:
        return "=e";
    case NVIMGCDCS_FLOAT64:
        return "=d";
    case NVIMGCDCS_BOOL:
        return "=?";*/
    default:
        break;
    }
}

static nvimgcdcsSampleDataType_t type_from_format_str(const std::string& typestr)
{
    int itemsize = py::dtype(typestr).itemsize();
    if (itemsize == 1) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;

    } else if (itemsize == 2) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
    } else if (itemsize == 4) {
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32;
    }
    return NVIMGCDCS_SAMPLE_DATA_TYPE_UNKNOWN;
}

//TODO move to Image.hpp
class Image
{
  public:
    Image()
        : image_(nullptr){};
    explicit Image(nvimgcdcsImage_t image)
        : image_(image)
    {
        nvimgcdcsImageInfo_t image_info;
        nvimgcdcsImageGetImageInfo(image_, &image_info);
        void* buffer;
        size_t buf_size;

        nvimgcdcsImageGetDeviceBuffer(image_, &buffer, &buf_size);

        ssize_t itemsize   = 1; //TODO
        ssize_t size       = buf_size;
        std::string format = format_str_from_type(image_info.sample_type);
        ssize_t ndim       = 3; //TODO
        std::vector<ssize_t> shape{
            image_info.image_height, image_info.image_width, image_info.num_components};
        std::vector<ssize_t> strides{
            static_cast<ssize_t>(image_info.component_info[0].pitch_in_bytes),
            static_cast<ssize_t>(1),
            static_cast<ssize_t>(
                image_info.component_info[0].pitch_in_bytes * image_info.image_height)};
        py::tuple strides_tuple = py::make_tuple(image_info.component_info[0].pitch_in_bytes, 1,
            image_info.component_info[0].pitch_in_bytes * image_info.image_height);

        buf_info_ = py::buffer_info(buffer, itemsize, format, ndim, shape, strides, false);
        try {
            py::tuple shape_tuple = py::make_tuple(
                image_info.image_height, image_info.image_width, image_info.num_components);

            // clang-format off
            cudaarrayinterface_ = py::dict
            {
                "shape"_a = shape_tuple,
                "strides"_a = strides_tuple,
                "typestr"_a = buf_info_.format,
                "data"_a = py::make_tuple(py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(buffer)), buf_info_.readonly),
                "version"_a = 2 
            };
            // clang-format on
        } catch (...) {
            throw;
        }
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
    int getNdim() const { return 3; } //TODO

    py::dict cuda_interface() const { return cudaarrayinterface_; }
    py::object shape() const { return cudaarrayinterface_["shape"]; }

    py::object dtype() const
    {
        nvimgcdcsImageInfo_t image_info;
        nvimgcdcsImageGetImageInfo(image_, &image_info);
        std::string format = format_str_from_type(image_info.sample_type);
        return py::dtype(format);
    }

    static Image* createImageFromFile(nvimgcdcsInstance_t instance, const char* file_name)
    {
        nvimgcdcsImage_t image;
        nvimgcdcsImRead(instance, &image, file_name);
        return new Image(image);
    }

    static Image* createImageFromPy(nvimgcdcsInstance_t instance, PyObject* o)
    {
        if (!o) {
            return nullptr;
        }
        py::object tmp{o, true /* borrowed */};

        if (hasattr(tmp, "__cuda_array_interface__")) {
            py::dict iface = tmp.attr("__cuda_array_interface__").cast<py::dict>();

            if (!iface.contains("shape") || !iface.contains("typestr") || !iface.contains("data") ||
                !iface.contains("version")) {
                return nullptr;
            }
            int version = iface["version"].cast<int>();
            if (version < 2) {
                return nullptr;
            }
            py::tuple tdata = iface["data"].cast<py::tuple>();
            void* buffer    = PyLong_AsVoidPtr(tdata[0].ptr());
            check_cuda_buffer(buffer);
            std::vector<long> vshape;
            py::tuple shape = iface["shape"].cast<py::tuple>();
            for (auto& o : shape) {
                vshape.push_back(o.cast<long>());
            }
            std::vector<int> vstrides;
            if (iface.contains("strides")) {
                py::object strides = iface["strides"];
                if (strides != py::none()) {
                    strides = strides.cast<py::tuple>();
                    for (auto& o : strides) {
                        vstrides.push_back(o.cast<int>());
                    }
                }
            }
            if (vshape.size() >= 2) {
                nvimgcdcsImageInfo_t image_info;
                image_info.image_height   = vshape[0];
                image_info.image_width    = vshape[1];
                image_info.num_components = (vshape.size() >= 3) ? vshape[2] : 1;
                std::string typestr       = iface["typestr"].cast<std::string>();
                image_info.sample_type    = type_from_format_str(typestr);
                size_t buffer_size        = 0;
                image_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_I_RGB;
                int pitch_in_bytes        = vstrides.size() > 0
                                                ? vstrides[0]
                                                : image_info.image_width * image_info.num_components;
                for (size_t c = 0; c < image_info.num_components; c++) {
                    image_info.component_info[c].component_width  = image_info.image_width;
                    image_info.component_info[c].component_height = image_info.image_height;
                    image_info.component_info[c].pitch_in_bytes   = pitch_in_bytes;
                    image_info.component_info[c].sample_type      = image_info.sample_type;
                    buffer_size +=
                        image_info.component_info[c].pitch_in_bytes * image_info.image_height;
                }

                nvimgcdcsImage_t image;
                nvimgcdcsImageCreate(instance, &image, &image_info);
                nvimgcdcsImageSetDeviceBuffer(image, buffer, buffer_size);
                return new Image(image);
            } else {
                return nullptr;
            }
        } else {
            return nullptr;
        }
    }

    nvimgcdcsImage_t image_;

  private:
    py::buffer_info buf_info_;
    py::dict cudaarrayinterface_;
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
        .def_property_readonly("__cuda_array_interface__", &Image::cuda_interface)
        .def_property_readonly("shape", &Image::shape)
        .def_property_readonly("width", &Image::getWidth)
        .def_property_readonly("height", &Image::getHeight)
        .def_property_readonly("ndim", &Image::getNdim)
        .def_property_readonly("dtype", &Image::dtype);

    m.def("imread", [](const char* file_name) -> Image* {
        return Image::createImageFromFile(module.instance_, file_name);
    }, "Loads an image from a specified file", "file_name"_a);
    m.def(
        "imwrite",
        [](const char* file_name, Image* image, const int* params = NULL) -> void {
            nvimgcdcsImWrite(module.instance_, image->image_, file_name, params);
        },
        "Saves an image to a specified file", "file_name"_a, "image"_a, "params"_a = NULL);
    m.def("asimage", [](py::handle src) -> Image* {
        return Image::createImageFromPy(module.instance_, src.ptr());
    });
}
