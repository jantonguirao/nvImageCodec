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

#include <filesystem>
#include <iostream>
#include "../src/file_ext_codec.h"
#include "error_handling.h"

namespace fs = std::filesystem;

namespace nvimgcdcs {

struct Encoder::EncoderDeleter
{
    void operator()(nvimgcdcsEncoder_t encoder) { nvimgcdcsEncoderDestroy(encoder); }
};

Encoder::Encoder(nvimgcdcsInstance_t instance, int device_id, const std::string& options)
    : encoder_(nullptr)
    , instance_(instance)
{
    nvimgcdcsEncoder_t encoder;
    nvimgcdcsEncoderCreate(instance, &encoder, device_id, options.c_str());
    encoder_ = std::shared_ptr<std::remove_pointer<nvimgcdcsEncoder_t>::type>(encoder, EncoderDeleter{});
}

Encoder::~Encoder()
{
}

py::bytes Encoder::encode(Image image, const std::string& codec, int cuda_stream)
{
    std::vector<Image> images{image};

    std::vector<py::bytes> data_list = encode(images, codec,  cuda_stream);
    if (data_list.size() == 1)
        return data_list[0];
    else
        return py::bytes(nullptr);
}

void Encoder::encode(const std::string& file_name, Image image, const std::string& codec, int cuda_stream)
{
    std::vector<Image> images{image};
    std::vector<std::string> file_names{file_name};

    encode(file_names, images, codec, cuda_stream);
}

void Encoder::encode(const std::vector<Image>& images, int cuda_stream,
    std::function<void(const nvimgcdcsImageInfo_t& out_image_info, nvimgcdcsCodeStream_t* code_stream)> create_code_stream)
{
    std::vector<nvimgcdcsCodeStream_t> code_streams(images.size());
    std::vector<nvimgcdcsImage_t> int_images(images.size());

    nvimgcdcsEncodeParams_t encode_params{NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS, 0};

    //Defaults
    encode_params.quality = 95;
    encode_params.target_psnr = 50;
    encode_params.mct_mode = NVIMGCDCS_MCT_MODE_RGB;
    nvimgcdcsJpegImageInfo_t jpeg_image_info{NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
    nvimgcdcsJpeg2kEncodeParams_t jpeg2k_encode_params{NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, 0};
    nvimgcdcsJpegEncodeParams_t jpeg_encode_params{NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, 0};
    // if (codec_name == "jpeg2k") {
    jpeg2k_encode_params.type = NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
    jpeg2k_encode_params.stream_type = NVIMGCDCS_JPEG2K_STREAM_JP2;
    jpeg2k_encode_params.prog_order = NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL;
    jpeg2k_encode_params.num_resolutions = 5;
    jpeg2k_encode_params.code_block_w = 64;
    jpeg2k_encode_params.code_block_h = 64;
    jpeg2k_encode_params.irreversible = true;
    encode_params.next = &jpeg2k_encode_params;
    //   } else if (codec_name == "jpeg") {
    jpeg_encode_params.type = NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS;
    jpeg_image_info.encoding = NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT;
    jpeg_encode_params.optimized_huffman = false;
    encode_params.next = &jpeg_encode_params;
    //   }

    //fill_encode_params(params, &encode_params, &out_image_info, &device_id);

    for (uint32_t i = 0; i < images.size(); i++) {
        int_images[i] = images[i].getNvImgCdcsImage();

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        nvimgcdcsImageGetImageInfo(int_images[i], &image_info);

        nvimgcdcsImageInfo_t out_image_info(image_info);
        out_image_info.next = &jpeg_image_info;
        jpeg_image_info.next = image_info.next;

        create_code_stream(out_image_info, &code_streams[i]);
    }
    nvimgcdcsFuture_t encode_future;
    CHECK_NVIMGCDCS(
        nvimgcdcsEncoderEncode(encoder_.get(), int_images.data(), code_streams.data(), images.size(), &encode_params, &encode_future));
    nvimgcdcsFutureWaitForAll(encode_future);
    size_t status_size;
    nvimgcdcsFutureGetProcessingStatus(encode_future, nullptr, &status_size);
    std::vector<nvimgcdcsProcessingStatus_t> encode_status(status_size);
    nvimgcdcsFutureGetProcessingStatus(encode_future, &encode_status[0], &status_size);
    size_t skip_samples = 0;
    for (size_t i = 0; i < encode_status.size(); ++i) {
        if (encode_status[i] != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
            std::cerr << "Something went wrong during encoding image #" << i << " it will not be included in output" << std::endl;
            // data_list.erase(data_list.begin() + i - skip_samples);
            skip_samples++;
        }
    }
    nvimgcdcsFutureDestroy(encode_future);
    for (auto& cs : code_streams) {
        nvimgcdcsCodeStreamDestroy(cs);
    }
}

std::vector<py::bytes> Encoder::encode(const std::vector<Image>& images, const std::string& codec, int cuda_stream)
{
    std::vector<py::bytes> data_list;
    data_list.reserve(images.size());
    if (codec.empty()) {
        std::cerr << "Error: Unspecified codec." << std::endl;
        return data_list;
    }
    std::string codec_name = codec[0] == '.' ? file_ext_to_codec(codec) : codec;
    if (codec_name.empty()) {
        std::cerr << "Error: Unsupported codec." << std::endl;
        return data_list;
    }

    auto create_code_stream = [&](const nvimgcdcsImageInfo_t& out_image_info, nvimgcdcsCodeStream_t* code_stream) -> void {
        size_t buffer_size = out_image_info.buffer_size;
        PyObject* bytesObject = PyBytes_FromStringAndSize(nullptr, buffer_size);
        char* buffer = PyBytes_AsString(bytesObject);

        CHECK_NVIMGCDCS(nvimgcdcsCodeStreamCreateToHostMem(
            instance_, code_stream, (unsigned char*)buffer, buffer_size, codec_name.c_str(), &out_image_info));

        data_list.push_back(py::reinterpret_steal<py::object>(bytesObject));
    };

    encode(images, cuda_stream, create_code_stream);

    return data_list;
}

void Encoder::encode(
    const std::vector<std::string>& file_names, const std::vector<Image>& images, const std::string& codec, int cuda_stream)
{
    std::vector<nvimgcdcsCodeStream_t> code_streams(images.size());
    int i = 0;
    auto create_code_stream = [&](const nvimgcdcsImageInfo_t& out_image_info, nvimgcdcsCodeStream_t* code_stream) -> void {
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


        CHECK_NVIMGCDCS(
            nvimgcdcsCodeStreamCreateToFile(instance_, code_stream, file_names[i].c_str(), codec_name.c_str(), &out_image_info));
        i++;
    };

    encode(images, cuda_stream, create_code_stream);
}

void Encoder::exportToPython(py::module& m, nvimgcdcsInstance_t instance)
{
    py::class_<Encoder>(m, "Encoder")
        .def(py::init<>([instance](int device_id, const std::string& options) { return new Encoder(instance, device_id, options); }),
            R"pbdoc(
            Initialize encoder.

            Args:
                device_id: Device id to execute encoding on.
                options: Encoder specific options  

            )pbdoc",

            "device_id"_a = NVIMGCDCS_DEVICE_CURRENT, "options"_a = "")
        .def("encode", py::overload_cast<Image, const std::string&, int>(&Encoder::encode),
            R"pbdoc(
            Encode image to buffer.

            Args:
                image: Image to encode
                codec: String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a leading period e.g. '.jp2'.
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
            Returns:
                Buffer with compressed code stream
        )pbdoc",
            "image"_a, "codec"_a, "cuda_stream"_a = 0)
        .def("encode", py::overload_cast<const std::string&, Image, const std::string&, int>(&Encoder::encode),
            R"pbdoc(
            Encode image to file.

            Args:
                file_name: File name to save encoded code stream. 
                image: Image to encode
                codec (optional): String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a 
                                  leading period e.g. '.jp2'. If codec is not specified, it is deducted based on file extension. 
                                  If there is no extension by default 'jpeg' is choosen. 
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
            Returns:
                void
        )pbdoc",
            "file_name"_a, "image"_a, "codec"_a = "", "cuda_stream"_a = 0)
        .def("encode", py::overload_cast<const std::vector<Image>&, const std::string&, int>(&Encoder::encode),
            R"pbdoc(
            Encode batch of images to buffers.

            Args:
                images: List of images to encode
                codec: String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a leading period e.g. '.jp2'.
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
            Returns:
                List of buffers with compressed code streams
        )pbdoc",
            "images"_a, "codec"_a, "cuda_stream"_a = 0)
        .def("encode",
            py::overload_cast<const std::vector<std::string>&, const std::vector<Image>&, const std::string&, int>(&Encoder::encode),
            R"pbdoc(
            Encode batch of images to files.

            Args:
                images: List of images to encode
                file_names: List of file names to save encoded code streams.
                codec (optional): String that defines the output format e.g.'jpeg2k'. When it is file extension it must include a 
                    leading period e.g. '.jp2'. If codec is not specified, it is deducted based on file extension. 
                    If there is no extension by default 'jpeg' is choosen. 
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.
            Returns:
                List of buffers with compressed code streams
        )pbdoc",
            "file_names"_a, "images"_a, "codec"_a = "", "cuda_stream"_a = 0);
}

} // namespace nvimgcdcs
