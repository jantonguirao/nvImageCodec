/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "image.h"

#include <iostream>

#include "error_handling.h"

namespace nvimgcdcs {

static std::string format_str_from_type(nvimgcdcsSampleDataType_t type)
{
    switch (type) {
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT8:
        return "|i1";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8:
        return "|u1";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT16:
        return "<i2";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16:
        return "<u2";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT32:
        return "<i4";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT32:
        return "<u4";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT64:
        return "<i8";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT64:
        return "<u8";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT16:
        return "<f2";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32:
        return "<f4";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT64:
        return "<f8";
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
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT8;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
    } else if (itemsize == 2) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT16;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT16;
    } else if (itemsize == 4) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT32;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT32;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32;
    } else if (itemsize == 8) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT64;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT64;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT64;
    }
    return NVIMGCDCS_SAMPLE_DATA_TYPE_UNKNOWN;
}

struct Image::BufferDeleter
{
    void operator()(unsigned char* buffer) { cudaFree(buffer); }
};
struct Image::ImageDeleter
{
    void operator()(nvimgcdcsImage_t image) { nvimgcdcsImageDestroy(image); }
};

Image::Image(nvimgcdcsImage_t image)
    : img_buffer_size_(0)
{
    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsImageGetImageInfo(image, &image_info);
    initCudaArrayInterface(&image_info);
    image_ = std::shared_ptr<std::remove_pointer<nvimgcdcsImage_t>::type>(image, ImageDeleter{});
}

Image::Image(nvimgcdcsInstance_t instance, nvimgcdcsImageInfo_t* image_info)
    : img_buffer_size_(0)
{
    if (image_info->buffer == nullptr) {
        unsigned char* buffer;
        CHECK_CUDA(cudaMalloc((void**)&buffer, image_info->buffer_size));
        img_buffer_ = std::shared_ptr<unsigned char>(buffer, BufferDeleter{});
        img_buffer_size_ = image_info->buffer_size;
        image_info->buffer = buffer;
    }

    nvimgcdcsImage_t image;
    CHECK_NVIMGCDCS(nvimgcdcsImageCreate(instance, &image, image_info));
    image_ = std::shared_ptr<std::remove_pointer<nvimgcdcsImage_t>::type>(image, ImageDeleter{});
    initCudaArrayInterface(image_info);
}

Image::Image(nvimgcdcsInstance_t instance, PyObject* o, intptr_t cuda_stream)
    : img_buffer_size_(0)
{
    if (!o) {
        throw std::runtime_error("Error: Object cannot be None");
    }
    py::object tmp = py::reinterpret_borrow<py::object>(o);

    if (!hasattr(tmp, "__cuda_array_interface__")) {
        throw std::runtime_error("Error: Object does not support __cuda_array_interface__");
    }

    py::dict iface = tmp.attr("__cuda_array_interface__").cast<py::dict>();

    if (!iface.contains("shape") || !iface.contains("typestr") || !iface.contains("data") || !iface.contains("version")) {
        throw std::runtime_error("Error: Unsupported __cuda_array_interface__ with missing field(s)");
    }

    int version = iface["version"].cast<int>();
    if (version < 2) {
        throw std::runtime_error("Error: Unsupported __cuda_array_interface__ with version < 2");
    }

    py::tuple tdata = iface["data"].cast<py::tuple>();
    void* buffer = PyLong_AsVoidPtr(tdata[0].ptr());
    check_cuda_buffer(buffer);
    std::vector<long> vshape;
    py::tuple shape = iface["shape"].cast<py::tuple>();
    for (auto& o : shape) {
        vshape.push_back(o.cast<long>());
    }
    if (vshape.size() < 3) {
        throw std::runtime_error("Error: Unsupported vshape");
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

    std::optional<intptr_t> stream = version >= 3 ? iface["stream"].cast<std::optional<intptr_t>>() : std::optional<intptr_t>();

    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    if (stream.has_value()) {
        if (*stream == 0) {
            throw std::runtime_error("Error: Invalid for stream to be 0");
        } else {
            image_info.cuda_stream = reinterpret_cast<cudaStream_t>(*stream);
        }
    }
    image_info.cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    bool is_interleaved = true; //TODO detect interleaved if we have HWC layout

    if (is_interleaved) {
        image_info.num_planes = 1;
        image_info.plane_info[0].height = vshape[0];
        image_info.plane_info[0].width = vshape[1];
        image_info.plane_info[0].num_channels = vshape[2];
    } else {
        image_info.num_planes = vshape[0];
        image_info.plane_info[0].height = vshape[1];
        image_info.plane_info[0].width = vshape[2];
        image_info.plane_info[0].num_channels = 1;
    }

    std::string typestr = iface["typestr"].cast<std::string>();
    auto sample_type = type_from_format_str(typestr);

    int bytes_per_element = static_cast<unsigned int>(sample_type) >> (8 + 3);

    image_info.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
    image_info.sample_format = is_interleaved ? NVIMGCDCS_SAMPLEFORMAT_I_RGB : NVIMGCDCS_SAMPLEFORMAT_P_RGB;
    image_info.chroma_subsampling = NVIMGCDCS_SAMPLING_444;

    int pitch_in_bytes = vstrides.size() > 1 ? (is_interleaved ? vstrides[0] : vstrides[1])
                                             : image_info.plane_info[0].width * image_info.plane_info[0].num_channels * bytes_per_element;
    size_t buffer_size = 0;
    for (size_t c = 0; c < image_info.num_planes; c++) {
        image_info.plane_info[c].width = image_info.plane_info[0].width;
        image_info.plane_info[c].height = image_info.plane_info[0].height;
        image_info.plane_info[c].row_stride = pitch_in_bytes;
        image_info.plane_info[c].sample_type = sample_type;
        image_info.plane_info[c].num_channels = image_info.plane_info[0].num_channels;
        buffer_size += image_info.plane_info[c].row_stride * image_info.plane_info[0].height;
    }
    image_info.buffer = buffer;
    image_info.buffer_size = buffer_size;
    image_info.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE;

    nvimgcdcsImage_t image;
    CHECK_NVIMGCDCS(nvimgcdcsImageCreate(instance, &image, &image_info));
    image_ = std::shared_ptr<std::remove_pointer<nvimgcdcsImage_t>::type>(image, ImageDeleter{});
    initCudaArrayInterface(&image_info);
}

void Image::initCudaArrayInterface(nvimgcdcsImageInfo_t* image_info)
{
    void* buffer = image_info->buffer;
    std::string format = format_str_from_type(image_info->plane_info[0].sample_type);
    std::vector<ssize_t> shape{image_info->num_planes, image_info->plane_info[0].height, image_info->plane_info[0].width};
    bool is_interleaved = static_cast<int>(image_info->sample_format) % 2 == 0 || image_info->num_planes == 1;
    try {
        int bytes_per_element = static_cast<unsigned int>(image_info->plane_info[0].sample_type) >> (8 + 3);
        py::tuple strides_tuple = is_interleaved ? py::make_tuple(image_info->plane_info[0].row_stride,
                                                       image_info->plane_info[0].num_channels * bytes_per_element, bytes_per_element)
                                                 : py::make_tuple(image_info->plane_info[0].row_stride * image_info->plane_info[0].height,
                                                       image_info->plane_info[0].row_stride, bytes_per_element);

        py::tuple shape_tuple =
            is_interleaved
                ? py::make_tuple(image_info->plane_info[0].height, image_info->plane_info[0].width, image_info->plane_info[0].num_channels)
                : py::make_tuple(image_info->num_planes, image_info->plane_info[0].height, image_info->plane_info[0].width);

        py::object strides = is_interleaved ? py::object(py::none()) : py::object(strides_tuple);
        py::object stream = image_info->cuda_stream ? py::int_((intptr_t)(image_info->cuda_stream)) : py::int_(1);
        // clang-format off
        cuda_array_interface_ = 
             py::dict {
                    "shape"_a = shape_tuple,
                    "strides"_a =  strides,
                    "typestr"_a = format,
                    "data"_a = py::make_tuple(py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(buffer)), false),
                    "version"_a = 3,
                    "stream"_a = stream
                };
        // clang-format on
    } catch (...) {
        throw std::runtime_error("Error: Unable to initialize __cuda_array_interface__");
    }
}

int Image::getWidth() const
{
    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsImageGetImageInfo(image_.get(), &image_info);
    return image_info.plane_info[0].width;
}
int Image::getHeight() const
{
    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsImageGetImageInfo(image_.get(), &image_info);
    return image_info.plane_info[0].height;
}
int Image::getNdim() const
{
    return 3;
}

py::dict Image::cuda_interface() const
{
    return cuda_array_interface_;
}
py::object Image::shape() const
{
    return cuda_array_interface_["shape"];
}

py::object Image::dtype() const
{
    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsImageGetImageInfo(image_.get(), &image_info);
    std::string format = format_str_from_type(image_info.plane_info[0].sample_type);
    return py::dtype(format);
}

nvimgcdcsImage_t Image::getNvImgCdcsImage() const
{
    return image_.get();
}

void Image::exportToPython(py::module& m)
{
    py::class_<Image>(m, "Image")
        .def_property_readonly("__cuda_array_interface__", &Image::cuda_interface)
        .def_property_readonly("shape", &Image::shape)
        .def_property_readonly("width", &Image::getWidth)
        .def_property_readonly("height", &Image::getHeight)
        .def_property_readonly("ndim", &Image::getNdim)
        .def_property_readonly("dtype", &Image::dtype);
}

} // namespace nvimgcdcs
