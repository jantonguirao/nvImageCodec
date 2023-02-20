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
        instance_create_info.load_extension_modules  = true;
        instance_create_info.default_debug_messenger = true;
        instance_create_info.message_severity        = NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL |
                                                NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR |
                                                NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING;
        instance_create_info.message_type = NVIMGCDCS_DEBUG_MESSAGE_TYPE_ALL;
        instance_create_info.num_cpu_threads = 10;
        nvimgcdcsInstanceCreate(&instance_, instance_create_info);
    }
    ~Module() { nvimgcdcsInstanceDestroy(instance_); }

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
    return "";
}

static nvimgcdcsSampleDataType_t type_from_format_str(const std::string& typestr)
{
    pybind11::ssize_t itemsize = py::dtype(typestr).itemsize();
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
            image_info.num_components, image_info.image_height, image_info.image_width};
        bool is_interleaved = static_cast<int>(image_info.sample_format) % 2 == 0;

        py::tuple strides_tuple = py::make_tuple(
            is_interleaved
                ? 1
                : image_info.component_info[0].device_pitch_in_bytes * image_info.image_height,
            image_info.component_info[0].device_pitch_in_bytes, is_interleaved ? 3 : 1);

        try {
            //TODO interleaved
            py::tuple shape_tuple = py::make_tuple(
                image_info.num_components, image_info.image_height, image_info.image_width);

            // clang-format off
            // TODO when strides none
            //     cudaarrayinterface_ = py::dict
            //     {
            //         "shape"_a = shape_tuple,
            //         "strides"_a = py::none(),  
            //         "typestr"_a = format,
            //         "data"_a = py::make_tuple(py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(buffer)), false),
            //         "version"_a = 2 
            //     };
                cudaarrayinterface_ = py::dict
                {
                    "shape"_a = shape_tuple,
                    "strides"_a = strides_tuple,  
                    "typestr"_a = format,
                    "data"_a = py::make_tuple(py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(buffer)), false),
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

    static Image* createImageFromFile(
        nvimgcdcsInstance_t instance, const char* file_name, int flags)
    {
        nvimgcdcsImage_t image;
        nvimgcdcsImRead(instance, &image, file_name, flags);
        return new Image(image);
    }

    static Image* createImageFromPy(nvimgcdcsInstance_t instance, PyObject* o)
    {
        if (!o) {
            return nullptr;
        }
        py::object tmp = py::reinterpret_borrow<py::object>(o);

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
                if (!strides.is(py::none())) {
                    strides = strides.cast<py::tuple>();
                    for (auto& o : strides) {
                        vstrides.push_back(o.cast<int>());
                    }
                }
            }
            if (vshape.size() >= 3) {
                nvimgcdcsImageInfo_t image_info;
                image_info.num_components = vshape[0];
                image_info.image_height   = vshape[1];
                image_info.image_width    = vshape[2];
                std::string typestr       = iface["typestr"].cast<std::string>();
                image_info.sample_type    = type_from_format_str(typestr);
                size_t buffer_size        = 0;
                image_info.color_space    = NVIMGCDCS_COLORSPACE_SRGB;
                image_info.sample_format =
                    NVIMGCDCS_SAMPLEFORMAT_P_RGB; //NVIMGCDCS_SAMPLEFORMAT_I_RGB; //TODO add support for various formats
                image_info.sampling = NVIMGCDCS_SAMPLING_444;
                int pitch_in_bytes  = vstrides.size() > 1
                                          ? vstrides[1]
                                          : image_info.image_width; //*image_info.num_components;
                for (size_t c = 0; c < image_info.num_components; c++) {
                    image_info.component_info[c].component_width       = image_info.image_width;
                    image_info.component_info[c].component_height      = image_info.image_height;
                    image_info.component_info[c].device_pitch_in_bytes = pitch_in_bytes;
                    image_info.component_info[c].sample_type           = image_info.sample_type;
                    buffer_size += image_info.component_info[c].device_pitch_in_bytes *
                                   image_info.image_height;
                }

                nvimgcdcsImage_t image;
                nvimgcdcsImageCreate(instance, &image);
                nvimgcdcsImageSetImageInfo(image, &image_info);
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

    m.def(
        "imread",
        [](const char* file_name, int flags) -> Image* {
            return Image::createImageFromFile(module.instance_, file_name, flags);
        },
        "Loads an image from a specified file", "file_name"_a,
        "flags"_a = static_cast<int>(NVIMGCDCS_IMREAD_COLOR));
    m.def(
        "imwrite",
        [](const char* file_name, Image* image, const std::vector<int>& params) -> void {
            std::vector<int> params_with_ending_zero = params;
            params_with_ending_zero.push_back(0);
            nvimgcdcsImWrite(
                module.instance_, image->image_, file_name, params_with_ending_zero.data());
        },
        "Saves an image to a specified file", "file_name"_a, "image"_a,
        "params"_a = std::vector<int>());
    m.def("asimage", [](py::handle src) -> Image* {
        return Image::createImageFromPy(module.instance_, src.ptr());
    });

    m.attr("NVIMGCDCS_IMREAD_GRAYSCALE") =
        static_cast<int>(NVIMGCDCS_IMREAD_GRAYSCALE); // do not convert to RGB
    //for jpeg with 4 color components assumes CMYK colorspace and converts to RGB
    //for Jpeg2k and 422/420 chroma subsampling enable conversion to RGB
    m.attr("NVIMGCDCS_IMREAD_COLOR") = static_cast<int>(NVIMGCDCS_IMREAD_COLOR);
    m.attr("NVIMGCDCS_IMREAD_IGNORE_ORIENTATION") =
        static_cast<int>(NVIMGCDCS_IMREAD_IGNORE_ORIENTATION); //Ignore orientation from Exif;
    m.attr("NVIMGCDCS_IMWRITE_JPEG_QUALITY") =
        static_cast<int>(NVIMGCDCS_IMWRITE_JPEG_QUALITY); // 0-100 default 95
    m.attr("NVIMGCDCS_IMWRITE_JPEG_PROGRESSIVE") =
        static_cast<int>(NVIMGCDCS_IMWRITE_JPEG_PROGRESSIVE);
    m.attr("NVIMGCDCS_IMWRITE_JPEG_OPTIMIZE") =
        static_cast<int>(NVIMGCDCS_IMWRITE_JPEG_OPTIMIZE); //optimized_huffman
    m.attr("NVIMGCDCS_IMWRITE_JPEG_SAMPLING_FACTOR") =
        static_cast<int>(NVIMGCDCS_IMWRITE_JPEG_SAMPLING_FACTOR);
    m.attr("NVIMGCDCS_IMWRITE_JPEG2K_TARGET_PSNR") =
        static_cast<int>(NVIMGCDCS_IMWRITE_JPEG2K_TARGET_PSNR); // default 50
    m.attr("NVIMGCDCS_IMWRITE_JPEG2K_NUM_DECOMPS") =
        static_cast<int>(NVIMGCDCS_IMWRITE_JPEG2K_NUM_DECOMPS); // num_decomps default 5
    m.attr("NVIMGCDCS_IMWRITE_JPEG2K_CODE_BLOCK_SIZE") = static_cast<int>(
        NVIMGCDCS_IMWRITE_JPEG2K_CODE_BLOCK_SIZE); // code_block_w code_block_h (default 64 64)
    m.attr("NVIMGCDCS_IMWRITE_JPEG2K_REVERSIBLE") =
        static_cast<int>(NVIMGCDCS_IMWRITE_JPEG2K_REVERSIBLE);
    m.attr("NVIMGCDCS_IMWRITE_MCT_MODE") = static_cast<int>(
        NVIMGCDCS_IMWRITE_MCT_MODE); // nvimgcdcsMctMode_t value (default NVIMGCDCS_MCT_MODE_RGB )
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_444") =
        static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_444);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_422") =
        static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_422);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_420") =
        static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_420);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_440") =
        static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_440);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_411") =
        static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_411);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410") =
        static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410") =
        static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_GRAY") =
        static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_GRAY);
}