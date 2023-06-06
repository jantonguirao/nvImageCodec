/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "decoder.h"

#include <iostream>
#include "error_handling.h"

namespace nvimgcdcs {

struct Decoder::DecoderDeleter
{
    void operator()(nvimgcdcsDecoder_t decoder) {
        nvimgcdcsDecoderDestroy(decoder);        
    }
};

Decoder::Decoder(nvimgcdcsInstance_t instance, int device_id, const std::string& options)
    : decoder_(nullptr)
    , instance_(instance)
{
    nvimgcdcsDecoder_t decoder;
    nvimgcdcsDecoderCreate(instance, &decoder, device_id, options.c_str());
    decoder_ = std::shared_ptr<std::remove_pointer<nvimgcdcsDecoder_t>::type>(decoder, DecoderDeleter{});
}

Decoder::~Decoder()
{
}

std::vector<Image> Decoder::decode(const std::vector<std::string>& data_list)
{
    std::vector<nvimgcdcsCodeStream_t> code_streams(data_list.size());
    std::vector<nvimgcdcsImage_t> images(data_list.size());
    std::vector<Image> py_images;
    py_images.reserve(data_list.size());

    nvimgcdcsDecodeParams_t decode_params{NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, 0};
    decode_params.enable_color_conversion = true;
    size_t skip_samples = 0;
    for (uint32_t i = 0; i < data_list.size(); i++) {
        CHECK_NVIMGCDCS(
            nvimgcdcsCodeStreamCreateFromHostMem(instance_, &code_streams[i], (unsigned char*)data_list[i].data(), data_list[i].size()));

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        CHECK_NVIMGCDCS(nvimgcdcsCodeStreamGetImageInfo(code_streams[i], &image_info));

        if (image_info.num_planes > NVIMGCDCS_MAX_NUM_PLANES) {
            std::cerr << "Warning: Num Components > " << NVIMGCDCS_MAX_NUM_PLANES
                     << "not supported.  It will not be included in output" << std::endl;

            skip_samples++;
            continue;
        }

        char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
        CHECK_NVIMGCDCS(nvimgcdcsCodeStreamGetCodecName(code_streams[i], codec_name));

        int bytes_per_element = static_cast<unsigned int>(image_info.plane_info[0].sample_type) >> (8 + 3);

        //Decode to format
        image_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        image_info.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        image_info.chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;

        bool swap_wh = decode_params.enable_orientation && ((image_info.orientation.rotated / 90) % 2);
        if (swap_wh) {
            std::swap(image_info.plane_info[0].height, image_info.plane_info[0].width);
        }

        size_t device_pitch_in_bytes = image_info.plane_info[0].width * bytes_per_element;

        for (uint32_t c = 0; c < image_info.num_planes; ++c) {
            image_info.plane_info[c].height = image_info.plane_info[0].height;
            image_info.plane_info[c].width = image_info.plane_info[0].width;
            image_info.plane_info[c].row_stride = device_pitch_in_bytes;
        }

        image_info.buffer_size = image_info.plane_info[0].row_stride * image_info.plane_info[0].height * image_info.num_planes;
        image_info.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE;

        py_images.emplace_back(instance_, &image_info);
        images[i - skip_samples] = py_images[i - skip_samples].getNvImgCdcsImage();
    }
    nvimgcdcsFuture_t decode_future;
    CHECK_NVIMGCDCS(nvimgcdcsDecoderDecode(decoder_.get(), code_streams.data(), images.data(), data_list.size(), &decode_params, &decode_future));
    nvimgcdcsFutureWaitForAll(decode_future);
    size_t status_size;
    nvimgcdcsFutureGetProcessingStatus(decode_future, nullptr, &status_size);
    std::vector<nvimgcdcsProcessingStatus_t> decode_status(status_size);
    nvimgcdcsFutureGetProcessingStatus(decode_future, &decode_status[0], &status_size);
    skip_samples = 0;
    for (size_t i = 0; i < decode_status.size(); ++i) {
        if (decode_status[i] != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
            std::cerr << "Something went wrong during decoding image #" << i << " it will not be included in output" <<std::endl;
            py_images.erase(py_images.begin() + i - skip_samples);
            skip_samples++;
        }
    }
    nvimgcdcsFutureDestroy(decode_future);
    for (auto& cs : code_streams) {
        nvimgcdcsCodeStreamDestroy(cs);
    }

    return py_images;
}

void Decoder::exportToPython(py::module& m, nvimgcdcsInstance_t instance)
{
    py::class_<Decoder>(m, "Decoder")
        .def(py::init<>([instance](int device_id, const std::string& options) { return new Decoder(instance, device_id, options); }),
            R"pbdoc(
            Initialize decoder.

            Args:
                device_id: Device id to execute decoding on.
                options: Decoder specific options e.g.: "nvjpeg:fancy_upsampling=1"

            )pbdoc",
            "device_id"_a = NVIMGCDCS_DEVICE_CURRENT, 
            "options"_a = ":fancy_upsampling=0")
        .def("decode", &Decoder::decode, R"pbdoc(
            Executes batch decoding.

            Args:
                data_list: List of buffers with code streams to decode.
            
            Returns:
                List of decoded Images

        )pbdoc", 
        "data_list"_a);
}

} // namespace nvimgcdcs
