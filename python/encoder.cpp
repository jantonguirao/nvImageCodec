/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "encoder.h"

#include <string.h>
#include <filesystem>
#include <iostream>
#include "../src/file_ext_codec.h"
#include "backend.h"
#include "error_handling.h"

namespace fs = std::filesystem;

namespace nvimgcdcs {

struct Encoder::EncoderDeleter
{
    void operator()(nvimgcdcsEncoder_t encoder) { nvimgcdcsEncoderDestroy(encoder); }
};

Encoder::Encoder(nvimgcdcsInstance_t instance, int device_id, std::optional<std::vector<Backend>> backends, const std::string& options)
    : encoder_(nullptr)
    , instance_(instance)
{
    std::vector<nvimgcdcsBackend_t> nvimgcds_backends(backends.has_value() ? backends.value().size() : 0);
    if (backends.has_value()) {
        for (size_t i = 0; i < backends.value().size(); ++i) {
            nvimgcds_backends[i] = backends.value()[i].backend_;
        }
    }

    auto backends_ptr = nvimgcds_backends.size() ? nvimgcds_backends.data() : nullptr;
    nvimgcdcsEncoder_t encoder;
    nvimgcdcsEncoderCreate(instance, &encoder, device_id, nvimgcds_backends.size(), backends_ptr, options.c_str());
    encoder_ = std::shared_ptr<std::remove_pointer<nvimgcdcsEncoder_t>::type>(encoder, EncoderDeleter{});
}

Encoder::Encoder(nvimgcdcsInstance_t instance, int device_id, std::optional<std::vector<nvimgcdcsBackendKind_t>> backend_kinds,
    const std::string& options)
    : encoder_(nullptr)
    , instance_(instance)
{
    std::vector<nvimgcdcsBackend_t> nvimgcds_backends(backend_kinds.has_value() ? backend_kinds.value().size() : 0);
    if (backend_kinds.has_value()) {
        for (size_t i = 0; i < backend_kinds.value().size(); ++i) {
            nvimgcds_backends[i].kind = backend_kinds.value()[i];
            nvimgcds_backends[i].params = {NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS, nullptr, 1.0f};
        }
    }
    auto backends_ptr = nvimgcds_backends.size() ? nvimgcds_backends.data() : nullptr;
    nvimgcdcsEncoder_t encoder;
    nvimgcdcsEncoderCreate(instance, &encoder, device_id, nvimgcds_backends.size(), backends_ptr, options.c_str());
    encoder_ = std::shared_ptr<std::remove_pointer<nvimgcdcsEncoder_t>::type>(encoder, EncoderDeleter{});
}

Encoder::~Encoder()
{
}

py::bytes Encoder::encode(Image image, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<Image> images{image};

    std::vector<py::bytes> data_list = encode(images, codec, params, cuda_stream);
    if (data_list.size() == 1)
        return data_list[0];
    else
        return py::bytes(nullptr);
}

void Encoder::encode(
    const std::string& file_name, Image image, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<Image> images{image};
    std::vector<std::string> file_names{file_name};

    encode(file_names, images, codec, params, cuda_stream);
}

void Encoder::encode(const std::vector<Image>& images, std::optional<EncodeParams> params_opt, intptr_t cuda_stream,
    std::function<void(size_t i, nvimgcdcsImageInfo_t& out_image_info, nvimgcdcsCodeStream_t* code_stream)> create_code_stream,
    std::function<void(size_t i, bool skip_item, nvimgcdcsCodeStream_t code_stream)> post_encode_call_back)
{
    std::vector<nvimgcdcsCodeStream_t> code_streams(images.size());
    std::vector<nvimgcdcsImage_t> int_images(images.size());
    EncodeParams params = params_opt.has_value() ? params_opt.value() : EncodeParams();

    params.jpeg2k_encode_params_.nvimgcdcs_jpeg2k_encode_params_.next = nullptr;
    params.jpeg_encode_params_.nvimgcdcs_jpeg_encode_params_.next = &params.jpeg2k_encode_params_.nvimgcdcs_jpeg2k_encode_params_;
    params.encode_params_.next = &params.jpeg_encode_params_.nvimgcdcs_jpeg_encode_params_;

    for (size_t i = 0; i < images.size(); i++) {
        int_images[i] = images[i].getNvImgCdcsImage();

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        nvimgcdcsImageGetImageInfo(int_images[i], &image_info);

        nvimgcdcsImageInfo_t out_image_info(image_info);
        out_image_info.chroma_subsampling = params.chroma_subsampling_;
        out_image_info.color_spec = params.color_spec_;
        out_image_info.next = (void*)(&params.jpeg_encode_params_.nvimgcdcs_jpeg_image_info_);

        create_code_stream(i, out_image_info, &code_streams[i]);
    }
    nvimgcdcsFuture_t encode_future;
    CHECK_NVIMGCDCS(nvimgcdcsEncoderEncode(
        encoder_.get(), int_images.data(), code_streams.data(), images.size(), &params.encode_params_, &encode_future));
    nvimgcdcsFutureWaitForAll(encode_future);
    size_t status_size;
    nvimgcdcsFutureGetProcessingStatus(encode_future, nullptr, &status_size);
    std::vector<nvimgcdcsProcessingStatus_t> encode_status(status_size);
    nvimgcdcsFutureGetProcessingStatus(encode_future, &encode_status[0], &status_size);
    for (size_t i = 0; i < encode_status.size(); ++i) {
        if (encode_status[i] != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
            std::cerr << "Error: Something went wrong during encoding image #" << i << " it will not be included in output" << std::endl;
        }
        post_encode_call_back(i, encode_status[i] != NVIMGCDCS_PROCESSING_STATUS_SUCCESS, code_streams[i]);
    }
    nvimgcdcsFutureDestroy(encode_future);
    for (auto& cs : code_streams) {
        nvimgcdcsCodeStreamDestroy(cs);
    }
}

std::vector<py::bytes> Encoder::encode(
    const std::vector<Image>& images, const std::string& codec, std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<py::bytes> data_list;
    if (codec.empty()) {
        std::cerr << "Error: Unspecified codec." << std::endl;
        return data_list;
    }
    std::string codec_name = codec[0] == '.' ? file_ext_to_codec(codec) : codec;
    if (codec_name.empty()) {
        std::cerr << "Error: Unsupported codec." << std::endl;
        return data_list;
    }

    struct PyObjectWrap
    {
        unsigned char* getBuffer(size_t bytes)
        {
            ptr_ = PyBytes_FromStringAndSize(nullptr, bytes);
            return (unsigned char*)PyBytes_AsString(ptr_);
        }

        static unsigned char* resize_buffer_static(void* ctx, size_t bytes)
        {
            auto handle = reinterpret_cast<PyObjectWrap*>(ctx);
            return handle->getBuffer(bytes);
        }

        PyObject* ptr_;
    };

    std::vector<PyObjectWrap> py_objects(images.size());

    auto create_code_stream = [&](size_t i, nvimgcdcsImageInfo_t& out_image_info, nvimgcdcsCodeStream_t* code_stream) -> void {
        strcpy(out_image_info.codec_name, codec_name.c_str());
        CHECK_NVIMGCDCS(nvimgcdcsCodeStreamCreateToHostMem(
            instance_, code_stream, (void*)&py_objects[i], &PyObjectWrap::resize_buffer_static, &out_image_info));
    };

    data_list.reserve(images.size());
    auto post_encode_callback = [&](size_t i, bool skip_item, nvimgcdcsCodeStream_t code_stream) -> void {
        if (skip_item && py_objects[i].ptr_) {
            Py_DECREF(py_objects[i].ptr_);
        } else {
            data_list.push_back(py::reinterpret_steal<py::object>(py_objects[i].ptr_));
        }
    };

    encode(images, params, cuda_stream, create_code_stream, post_encode_callback);

    return data_list;
}

void Encoder::encode(const std::vector<std::string>& file_names, const std::vector<Image>& images, const std::string& codec,
    std::optional<EncodeParams> params, intptr_t cuda_stream)
{
    std::vector<nvimgcdcsCodeStream_t> code_streams(images.size());
    auto create_code_stream = [&](size_t i, nvimgcdcsImageInfo_t& out_image_info, nvimgcdcsCodeStream_t* code_stream) -> void {
        std::string codec_name{};

        if (codec.empty()) {
            auto file_extension = fs::path(file_names[i]).extension();
            codec_name = file_ext_to_codec(file_extension);
            if (codec_name.empty()) {
                std::cerr << "Warning: File '" << file_names[i] << "' without extension. As default choosing jpeg codec" << std::endl;
                codec_name = "jpeg";
            }
        } else {
            codec_name = codec[0] == '.' ? file_ext_to_codec(codec) : codec;
            if (codec_name.empty()) {
                std::cerr << "Warning: Unsupported codec.  As default choosing jpeg codec" << std::endl;
                codec_name = "jpeg";
            }
        }
        strcpy(out_image_info.codec_name, codec_name.c_str());
        CHECK_NVIMGCDCS(nvimgcdcsCodeStreamCreateToFile(instance_, code_stream, file_names[i].c_str(), &out_image_info));
    };
    auto post_encode_callback = [&](size_t i, bool skip_item, nvimgcdcsCodeStream_t code_stream) -> void {};
    encode(images, params, cuda_stream, create_code_stream, post_encode_callback);
}

void Encoder::exportToPython(py::module& m, nvimgcdcsInstance_t instance)
{
    py::class_<Encoder>(m, "Encoder")
        .def(py::init<>([instance](int device_id, std::optional<std::vector<Backend>> backends, const std::string& options) {
            return new Encoder(instance, device_id, backends, options);
        }),
            R"pbdoc(
            Initialize encoder.

            Args:
                device_id: Device id to execute encoding on.
                backends: List of allowed backends. If empty, all backends are allowed with default parameters.
                options: Encoder specific options.  

            )pbdoc",

            "device_id"_a = NVIMGCDCS_DEVICE_CURRENT, "backends"_a = py::none(), "options"_a = "")
        .def(py::init<>([instance](int device_id, std::optional<std::vector<nvimgcdcsBackendKind_t>> backend_kinds,
                            const std::string& options) { return new Encoder(instance, device_id, backend_kinds, options); }),
            R"pbdoc(
            Initialize encoder.

            Args:
                device_id: Device id to execute encoding on.
                backend_kinds: List of allowed backend kinds. If empty or None, all backends are allowed with default parameters.
                options: Encoder specific options.

            )pbdoc",
            "device_id"_a = NVIMGCDCS_DEVICE_CURRENT, "backend_kinds"_a = py::none(), "options"_a = ":fancy_upsampling=0")
        .def("encode", py::overload_cast<Image, const std::string&, std::optional<EncodeParams>, intptr_t>(&Encoder::encode),
            R"pbdoc(
            Encode image to buffer.

            Args:
                image: Image to encode
                codec: String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a leading period e.g. '.jp2'.
                params: Encode parameters.
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
            Returns:
                Buffer with compressed code stream
        )pbdoc",
            "image"_a, "codec"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("write",
            py::overload_cast<const std::string&, Image, const std::string&, std::optional<EncodeParams>, intptr_t>(&Encoder::encode),
            R"pbdoc(
            Encode image to file.

            Args:
                file_name: File name to save encoded code stream. 
                image: Image to encode
                codec (optional): String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a 
                                  leading period e.g. '.jp2'. If codec is not specified, it is deducted based on file extension. 
                                  If there is no extension by default 'jpeg' is choosen. 
                params: Encode parameters.
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
            Returns:
                void
        )pbdoc",
            "file_name"_a, "image"_a, "codec"_a = "", "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("encode",
            py::overload_cast<const std::vector<Image>&, const std::string&, std::optional<EncodeParams>, intptr_t>(&Encoder::encode),
            R"pbdoc(
            Encode batch of images to buffers.

            Args:
                images: List of images to encode
                codec: String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a leading period e.g. '.jp2'.
                params: Encode parameters.
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
            Returns:
                List of buffers with compressed code streams
        )pbdoc",
            "images"_a, "codec"_a, "params"_a = py::none(), "cuda_stream"_a = 0)
        .def("write",
            py::overload_cast<const std::vector<std::string>&, const std::vector<Image>&, const std::string&, std::optional<EncodeParams>,
                intptr_t>(&Encoder::encode),
            R"pbdoc(
            Encode batch of images to files.

            Args:
                images: List of images to encode
                file_names: List of file names to save encoded code streams.
                codec (optional): String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a 
                    leading period e.g. '.jp2'. If codec is not specified, it is deducted based on file extension. 
                    If there is no extension by default 'jpeg' is choosen. 
                params: Encode parameters.
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
            Returns:
                List of buffers with compressed code streams
        )pbdoc",
            "file_names"_a, "images"_a, "codec"_a = "", "params"_a = py::none(), "cuda_stream"_a = 0);
}

} // namespace nvimgcdcs
