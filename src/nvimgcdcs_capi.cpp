/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <nvimgcodecs.h>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include "code_stream.h"
#include "codec_registry.h"
#include "exception.h"
#include "icodec.h"
#include "idecode_state.h"
#include "iencode_state.h"
#include "iimage_decoder.h"
#include "iimage_encoder.h"
#include "image.h"
#include "iostream_factory.h"

#include "log.h"
#include "nvimgcodecs_director.h"
#include "plugin_framework.h"
#include "processing_results.h"

namespace fs = std::filesystem;

using namespace nvimgcdcs;

__inline__ nvimgcdcsStatus_t getCAPICode(Status status)
{
    nvimgcdcsStatus_t code = NVIMGCDCS_STATUS_SUCCESS;
    switch (status) {
    case STATUS_OK:
        code = NVIMGCDCS_STATUS_SUCCESS;
        break;
    case NOT_VALID_FORMAT_STATUS:
    case PARSE_STATUS:
    case BAD_FORMAT_STATUS:
        code = NVIMGCDCS_STATUS_BAD_CODESTREAM;
        break;
    case UNSUPPORTED_FORMAT_STATUS:
        code = NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
        break;
    case CUDA_CALL_ERROR:
        code = NVIMGCDCS_STATUS_EXECUTION_FAILED;
        break;
    case ALLOCATION_ERROR:
        code = NVIMGCDCS_STATUS_ALLOCATOR_FAILURE;
        break;
    case INTERNAL_ERROR:
        code = NVIMGCDCS_STATUS_INTERNAL_ERROR;
        break;
    case INVALID_PARAMETER:
        code = NVIMGCDCS_STATUS_INVALID_PARAMETER;
        break;
    default:
        code = NVIMGCDCS_STATUS_INTERNAL_ERROR;
        break;
    }
    return code;
}

#ifdef NDEBUG
//TODO TEMP!!! #define VERBOSE_ERRORS
#else
    #define VERBOSE_ERRORS
#endif

// TODO use Logger
// TODO move to separate file

#define NVIMGCDCSAPI_TRY try

#ifndef VERBOSE_ERRORS
    #define NVIMGCDCSAPI_CATCH(a)                                  \
        catch (const Exception& e)                                 \
        {                                                          \
            a = getCAPICode(e.status());                           \
        }                                                          \
        catch (...)                                                \
        {                                                          \
            std::cerr << "Unknown NVIMGCODECS error" << std::endl; \
            a = NVIMGCDCS_STATUS_INTERNAL_ERROR;                   \
        }
#else
    #define NVIMGCDCSAPI_CATCH(a)                                     \
        catch (const Exception& e)                                    \
        {                                                             \
            std::cerr << "Error status: " << e.status() << std::endl; \
            std::cerr << "Where: " << e.where() << std::endl;         \
            std::cerr << "Message: " << e.message() << std::endl;     \
            std::cerr << "What: " << e.what() << std::endl;           \
            a = getCAPICode(e.status());                              \
        }                                                             \
        catch (const std::runtime_error& e)                           \
        {                                                             \
            std::cerr << "Error: " << e.what() << std::endl;          \
            a = NVIMGCDCS_STATUS_INTERNAL_ERROR;                      \
        }                                                             \
        catch (...)                                                   \
        {                                                             \
            std::cerr << "Unknown NVIMGCODECS error" << std::endl;    \
            a = NVIMGCDCS_STATUS_INTERNAL_ERROR;                      \
        }
#endif

inline size_t sample_type_to_bytes_per_element(nvimgcdcsSampleDataType_t sample_type)
{
    return ((static_cast<unsigned int>(sample_type) & 0b11111110) + 7) / 8;
}

struct nvimgcdcsInstance
{
    nvimgcdcsInstance(nvimgcdcsInstanceCreateInfo_t create_info)
        : director_(create_info)
    {
    }
    NvImgCodecsDirector director_;
};

struct nvimgcdcsFuture
{
    std::unique_ptr<ProcessingResultsFuture> handle_;
};

struct nvimgcdcsDecoder
{
    nvimgcdcsInstance_t instance_;
    std::unique_ptr<IImageDecoder> image_decoder_;
};

struct nvimgcdcsEncoder
{
    nvimgcdcsInstance_t instance_;
    std::unique_ptr<IImageEncoder> image_encoder_;
};

struct nvimgcdcsDebugMessenger
{
    nvimgcdcsInstance_t instance_;
    nvimgcdcsDebugMessenger(const nvimgcdcsDebugMessengerDesc_t* desc)
        : debug_messenger_(desc)
    {
    }
    DebugMessenger debug_messenger_;
};

struct nvimgcdcsExtension
{
    nvimgcdcsInstance_t nvimgcdcs_instance_;
    nvimgcdcsExtension_t extension_ext_handle_;
};

struct nvimgcdcsImage
{

    explicit nvimgcdcsImage()
        : image_()
        , dev_image_buffer_(nullptr)
        , dev_image_buffer_size_(0)
    {
    }

    ~nvimgcdcsImage()
    {
        if (dev_image_buffer_) {
            nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
            NVIMGCDCSAPI_TRY
                {
                    CHECK_CUDA(cudaFree(dev_image_buffer_));
                }
            NVIMGCDCSAPI_CATCH(ret)
            if (ret != NVIMGCDCS_STATUS_SUCCESS) {
                //TODO log
            }
        }
    }
    nvimgcdcsInstance_t nvimgcdcs_instance_;
    Image image_;
    void* dev_image_buffer_;
    size_t dev_image_buffer_size_;
};

nvimgcdcsStatus_t nvimgcdcsInstanceCreate(nvimgcdcsInstance_t* instance, nvimgcdcsInstanceCreateInfo_t create_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    nvimgcdcsInstance_t nvimgcdcs = nullptr;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance);
            nvimgcdcs = new nvimgcdcsInstance(create_info);
            *instance = nvimgcdcs;
        }
    NVIMGCDCSAPI_CATCH(ret)

    if (ret != NVIMGCDCS_STATUS_SUCCESS) {
        if (nvimgcdcs) {
            delete nvimgcdcs;
        }
    }

    return ret;
}

nvimgcdcsStatus_t nvimgcdcsInstanceDestroy(nvimgcdcsInstance_t instance)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            delete instance;
        }
    NVIMGCDCSAPI_CATCH(ret)

    return ret;
}

nvimgcdcsStatus_t nvimgcdcsExtensionCreate(
    nvimgcdcsInstance_t instance, nvimgcdcsExtension_t* extension, nvimgcdcsExtensionDesc_t* extension_desc)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(extension_desc)
            nvimgcdcsExtension_t extension_ext_handle;
            ret = instance->director_.plugin_framework_.registerExtension(&extension_ext_handle, extension_desc);
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                *extension = new nvimgcdcsExtension();
                (*extension)->nvimgcdcs_instance_ = instance;
                (*extension)->extension_ext_handle_ = extension_ext_handle;
            }
        }
    NVIMGCDCSAPI_CATCH(ret)

    return ret;
}

nvimgcdcsStatus_t nvimgcdcsExtensionDestroy(nvimgcdcsExtension_t extension)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(extension)

            return extension->nvimgcdcs_instance_->director_.plugin_framework_.unregisterExtension(extension->extension_ext_handle_);
        }
    NVIMGCDCSAPI_CATCH(ret)

    return ret;
}

struct nvimgcdcsCodeStream
{
    explicit nvimgcdcsCodeStream(CodecRegistry* codec_registry)
        : code_stream_(codec_registry, std::make_unique<IoStreamFactory>())
    {
    }
    nvimgcdcs::CodeStream code_stream_;
};

static nvimgcdcsStatus_t nvimgcdcsStreamCreate(nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    nvimgcdcsCodeStream_t stream = nullptr;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle);
            stream = new nvimgcdcsCodeStream(&instance->director_.codec_registry_);
            *stream_handle = stream;
        }
    NVIMGCDCSAPI_CATCH(ret)

    if (ret != NVIMGCDCS_STATUS_SUCCESS) {
        if (stream) {
            delete stream;
        }
    }
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromFile(
    nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle, const char* file_name)
{
    nvimgcdcsStatus_t ret = nvimgcdcsStreamCreate(instance, stream_handle);

    NVIMGCDCSAPI_TRY
        {
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                (*stream_handle)->code_stream_.parseFromFile(std::string(file_name));
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromHostMem(
    nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle, const unsigned char* data, size_t size)
{
    nvimgcdcsStatus_t ret = nvimgcdcsStreamCreate(instance, stream_handle);

    NVIMGCDCSAPI_TRY
        {
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                (*stream_handle)->code_stream_.parseFromMem(data, size);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToFile(nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle, const char* file_name,
    const char* codec_name, const nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = nvimgcdcsStreamCreate(instance, stream_handle);
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle)
            CHECK_NULL(file_name)
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                (*stream_handle)->code_stream_.setOutputToFile(file_name, codec_name);
                (*stream_handle)->code_stream_.setImageInfo(image_info);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToHostMem(nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle,
    unsigned char* output_buffer, size_t length, const char* codec_name, const nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = nvimgcdcsStreamCreate(instance, stream_handle);
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle)
            CHECK_NULL(output_buffer)
            CHECK_NULL(length)
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                (*stream_handle)->code_stream_.setOutputToHostMem(output_buffer, length, codec_name);
                (*stream_handle)->code_stream_.setImageInfo(image_info);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamDestroy(nvimgcdcsCodeStream_t stream_handle)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle)
            delete stream_handle;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamGetImageInfo(nvimgcdcsCodeStream_t stream_handle, nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle)
            CHECK_NULL(image_info)
            return stream_handle->code_stream_.getImageInfo(image_info);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamSetImageInfo(nvimgcdcsCodeStream_t stream_handle, nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle)
            CHECK_NULL(image_info)
            stream_handle->code_stream_.setImageInfo(image_info);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamGetCodecName(nvimgcdcsCodeStream_t stream_handle, char* codec_name)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle)
            CHECK_NULL(codec_name)
            std::string codec_name_ = stream_handle->code_stream_.getCodecName();
#ifdef WIN32
            strcpy_s(codec_name, NVIMGCDCS_MAX_CODEC_NAME_SIZE, codec_name_.c_str());
#else
            strcpy(codec_name, codec_name_.c_str());
#endif
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderCreate(nvimgcdcsInstance_t instance, nvimgcdcsDecoder_t* decoder, int device_id)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(decoder)
            if (device_id == -1)
                CHECK_CUDA(cudaGetDevice(&device_id));
            std::unique_ptr<IImageDecoder> image_decoder = instance->director_.createGenericDecoder(device_id);
            *decoder = new nvimgcdcsDecoder();
            (*decoder)->image_decoder_ = std::move(image_decoder);
            (*decoder)->instance_ = instance;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderDestroy(nvimgcdcsDecoder_t decoder)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            delete decoder;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderDecode(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStream_t* streams, nvimgcdcsImage_t* images,
    int batch_size, nvimgcdcsDecodeParams_t* params, nvimgcdcsFuture_t* future)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(streams)
            CHECK_NULL(images)
            CHECK_NULL(params)
            CHECK_NULL(future)

            std::vector<nvimgcdcs::ICodeStream*> internal_code_streams;
            std::vector<nvimgcdcs::IImage*> internal_images;

            for (int i = 0; i < batch_size; ++i) {
                internal_code_streams.push_back(&streams[i]->code_stream_);
                internal_images.push_back(&images[i]->image_);
            }
            *future = new nvimgcdcsFuture();

            (*future)->handle_ =
                std::move(decoder->image_decoder_->decode(nullptr, internal_code_streams, internal_images, params));
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsImageCreate(nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image, const nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            CHECK_NULL(instance)
            CHECK_NULL(image_info)
            CHECK_NULL(image_info->buffer)
            if (image_info->buffer_kind == NVIMGCDCS_IMAGE_BUFFER_KIND_UNKNOWN ||
                image_info->buffer_kind == NVIMGCDCS_IMAGE_BUFFER_KIND_UNSUPPORTED) {
                NVIMGCDCS_LOG_ERROR("Unknown or unsupported buffer kind");
                return NVIMGCDCS_STATUS_INVALID_PARAMETER;
            }

            *image = new nvimgcdcsImage();
            (*image)->image_.setImageInfo(image_info);
            (*image)->nvimgcdcs_instance_ = instance;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsImageDestroy(nvimgcdcsImage_t image)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            delete image;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsImageGetImageInfo(nvimgcdcsImage_t image, nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            image->image_.getImageInfo(image_info);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderCreate(nvimgcdcsInstance_t instance, nvimgcdcsEncoder_t* encoder, int device_id)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(encoder)
            if (device_id == -1)
                CHECK_CUDA(cudaGetDevice(&device_id));
            std::unique_ptr<IImageEncoder> image_encoder = instance->director_.createGenericEncoder(device_id);
            *encoder = new nvimgcdcsEncoder();
            (*encoder)->image_encoder_ = std::move(image_encoder);
            (*encoder)->instance_ = instance;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncoderDestroy(nvimgcdcsEncoder_t encoder)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encoder)
            delete encoder;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncoderEncode(nvimgcdcsEncoder_t encoder, nvimgcdcsImage_t* images, nvimgcdcsCodeStream_t* streams,
    int batch_size, nvimgcdcsEncodeParams_t* params, nvimgcdcsFuture_t* future)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(streams)
            CHECK_NULL(images)
            CHECK_NULL(params)
            CHECK_NULL(future)

            std::vector<nvimgcdcs::ICodeStream*> internal_code_streams;
            std::vector<nvimgcdcs::IImage*> internal_images;

            for (int i = 0; i < batch_size; ++i) {
                internal_code_streams.push_back(&streams[i]->code_stream_);
                internal_images.push_back(&images[i]->image_);
            }

            *future = new nvimgcdcsFuture();

            (*future)->handle_ = std::move(encoder->image_encoder_->encode(nullptr, internal_images, internal_code_streams, params));
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

static void fill_decode_params(const int* params, nvimgcdcsDecodeParams_t* decode_params, int* device_id)
{
    const int* param = params;
    while (param && *param) {
        NVIMGCDCS_LOG_TRACE("imwread param: " << *param);
        switch (*param) {
        case NVIMGCDCS_IMREAD_COLOR: {
            decode_params->enable_color_conversion = true;
            break;
        }
        case NVIMGCDCS_IMREAD_IGNORE_ORIENTATION: {
            decode_params->enable_orientation = false;
            break;
        }
        case NVIMGCDCS_IMREAD_DEVICE_ID: {
            param++;
            *device_id = *param;
            break;
        }
        default:
            break;
        };
        param++;
    }
}

nvimgcdcsStatus_t nvimgcdcsImRead(nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image, const char* file_name, const int* params)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(image)
            CHECK_NULL(file_name)

            nvimgcdcsCodeStream_t code_stream;
            nvimgcdcsCodeStreamCreateFromFile(instance, &code_stream, file_name);
            nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};;
            nvimgcdcsCodeStreamGetImageInfo(code_stream, &image_info);
            char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
            nvimgcdcsCodeStreamGetCodecName(code_stream, codec_name);

            int bytes_per_element = sample_type_to_bytes_per_element(image_info.plane_info[0].sample_type);

            // Define  requested output
            image_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
            image_info.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
            size_t device_pitch_in_bytes = image_info.plane_info[0].width * bytes_per_element;
            image_info.buffer_size = device_pitch_in_bytes * image_info.plane_info[0].height * image_info.num_planes;
            image_info.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
            CHECK_CUDA(cudaMalloc(&image_info.buffer, image_info.buffer_size));
            for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                image_info.plane_info[c].height = image_info.plane_info[0].height;
                image_info.plane_info[c].width = image_info.plane_info[0].width;
                image_info.plane_info[c].row_stride = device_pitch_in_bytes;
            }

            nvimgcdcsDecodeParams_t decode_params{NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, 0};
            // Defaults
            decode_params.enable_color_conversion = false;
            decode_params.enable_orientation = true;
            int device_id = NVIMGCDCS_DEVICE_CURRENT;
            fill_decode_params(params, &decode_params, &device_id);

            nvimgcdcsImageCreate(instance, image, &image_info);
            (*image)->dev_image_buffer_ = image_info.buffer;
            (*image)->dev_image_buffer_size_ = image_info.buffer_size;

            nvimgcdcsDecoder_t decoder;
            nvimgcdcsDecoderCreate(instance, &decoder, device_id);

            nvimgcdcsFuture_t future;
            nvimgcdcsDecoderDecode(decoder, &code_stream, image, 1, &decode_params, &future);
            nvimgcdcsProcessingStatus_t decode_status;
            size_t size;
            nvimgcdcsFutureGetProcessingStatus(future, &decode_status, &size);
            if (decode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
                NVIMGCDCS_LOG_ERROR("Something went wrong during decoding");
                ret = NVIMGCDCS_STATUS_EXECUTION_FAILED;
            }

            nvimgcdcsFutureDestroy(future);

            nvimgcdcsDecoderDestroy(decoder);
            nvimgcdcsCodeStreamDestroy(code_stream);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

static std::map<std::string, std::string> ext2codec = {{".bmp", "bmp"}, {".j2c", "jpeg2k"}, {".j2k", "jpeg2k"}, {".jp2", "jpeg2k"},
    {".tiff", "tiff"}, {".tif", "tiff"}, {".jpg", "jpeg"}, {".jpeg", "jpeg"}, {".ppm", "pxm"}, {".pgm", "pxm"}, {".pbm", "pxm"}};

static void fill_encode_params(const int* params, nvimgcdcsEncodeParams_t* encode_params, nvimgcdcsImageInfo_t* image_info, int* device_id)
{
    nvimgcdcsJpegEncodeParams_t* jpeg_encode_params = static_cast<nvimgcdcsJpegEncodeParams_t*>(encode_params->next);
    nvimgcdcsJpeg2kEncodeParams_t* jpeg2k_encode_params = static_cast<nvimgcdcsJpeg2kEncodeParams_t*>(encode_params->next);

    const int* param = params;
    while (param && *param) {
        NVIMGCDCS_LOG_TRACE("imwrite param: " << *param);
        switch (*param) {
        case NVIMGCDCS_IMWRITE_JPEG_QUALITY: {
            param++;
            int quality = *param;
            encode_params->quality = static_cast<float>(quality);
            NVIMGCDCS_LOG_TRACE("imwrite param: quality:" << *param);
            break;
        }
        case NVIMGCDCS_IMWRITE_JPEG_PROGRESSIVE: {
            jpeg_encode_params->encoding = NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;
            break;
        }
        case NVIMGCDCS_IMWRITE_JPEG_OPTIMIZE: {
            jpeg_encode_params->optimized_huffman = true;
            break;
        }
        case NVIMGCDCS_IMWRITE_JPEG_SAMPLING_FACTOR: {
            param++;
            NVIMGCDCS_LOG_DEBUG("imwrite param: sampling factor:" << *param);
            nvimgcdcsImWriteSamplingFactor_t sampling_factor = static_cast<nvimgcdcsImWriteSamplingFactor_t>(*param);
            std::map<nvimgcdcsImWriteSamplingFactor_t, nvimgcdcsChromaSubsampling_t> sf2css = {
                {NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_444, NVIMGCDCS_SAMPLING_444},
                {NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_420, NVIMGCDCS_SAMPLING_420},
                {NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_440, NVIMGCDCS_SAMPLING_440},
                {NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_422, NVIMGCDCS_SAMPLING_422},
                {NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_411, NVIMGCDCS_SAMPLING_411},
                {NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410, NVIMGCDCS_SAMPLING_410},
                {NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_GRAY, NVIMGCDCS_SAMPLING_GRAY},
                {NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410V, NVIMGCDCS_SAMPLING_410V}};

            auto it = sf2css.find(sampling_factor);
            if (it != sf2css.end()) {
                image_info->chroma_subsampling = it->second;
            } else {
                assert(!"MISSING CHROMA SUBSAMPLING VALUE");
            }
            break;
        }
        case NVIMGCDCS_IMWRITE_JPEG2K_TARGET_PSNR: {
            param++;
            int target_psnr = *param;
            memcpy(&encode_params->target_psnr, &target_psnr, sizeof(float));
            break;
        }
        case NVIMGCDCS_IMWRITE_JPEG2K_NUM_DECOMPS: {
            param++;
            jpeg2k_encode_params->num_resolutions = *param;
            break;
        }
        case NVIMGCDCS_IMWRITE_JPEG2K_CODE_BLOCK_SIZE: {
            param++;
            jpeg2k_encode_params->code_block_w = *param;
            param++;
            jpeg2k_encode_params->code_block_h = *param;
            break;
        }
        case NVIMGCDCS_IMWRITE_JPEG2K_REVERSIBLE: {
            jpeg2k_encode_params->irreversible = false;
            break;
        }
        case NVIMGCDCS_IMWRITE_JPEG2K_PROG_ORDER: {
            param++;
            jpeg2k_encode_params->prog_order = static_cast<nvimgcdcsJpeg2kProgOrder_t>(*param);
            break;
        }
        case NVIMGCDCS_IMWRITE_MCT_MODE: {
            param++;
            encode_params->mct_mode = static_cast<nvimgcdcsMctMode_t>(*param);
            break;
        }
        case NVIMGCDCS_IMWRITE_DEVICE_ID: {
            param++;
            *device_id = *param;
            break;
        }
        default:
            break;
        };
        param++;
    }
}

nvimgcdcsStatus_t nvimgcdcsImWrite(nvimgcdcsInstance_t instance, nvimgcdcsImage_t image, const char* file_name, const int* params)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(image)
            CHECK_NULL(file_name)

            nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
            nvimgcdcsImageGetImageInfo(image, &image_info);
            fs::path file_path(file_name);

            std::string codec_name = "bmp";
            if (file_path.has_extension()) {
                std::string extension = file_path.extension().string();
                auto it = ext2codec.find(extension);
                if (it != ext2codec.end()) {
                    codec_name = it->second;
                }
            }

            if (image_info.chroma_subsampling == NVIMGCDCS_SAMPLING_NONE || image_info.chroma_subsampling == NVIMGCDCS_SAMPLING_UNSUPPORTED)
                image_info.chroma_subsampling = NVIMGCDCS_SAMPLING_444;
            nvimgcdcsEncodeParams_t encode_params{NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS, 0};
            //Defaults
            encode_params.quality = 95;
            encode_params.target_psnr = 50;
            encode_params.mct_mode = NVIMGCDCS_MCT_MODE_RGB;

            nvimgcdcsJpeg2kEncodeParams_t jpeg2k_encode_params{};
            nvimgcdcsJpegEncodeParams_t jpeg_encode_params{};
            if (codec_name == "jpeg2k") {
                jpeg2k_encode_params.type = NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
                jpeg2k_encode_params.stream_type =
                    file_path.extension().string() == ".jp2" ? NVIMGCDCS_JPEG2K_STREAM_JP2 : NVIMGCDCS_JPEG2K_STREAM_J2K;
                jpeg2k_encode_params.prog_order = NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL; 
                jpeg2k_encode_params.num_resolutions = 5;
                jpeg2k_encode_params.code_block_w = 64;
                jpeg2k_encode_params.code_block_h = 64;
                jpeg2k_encode_params.irreversible = true;
                encode_params.next = &jpeg2k_encode_params;
            } else if (codec_name == "jpeg") {
                jpeg_encode_params.type = NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS;
                jpeg_encode_params.encoding = NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT;
                jpeg_encode_params.optimized_huffman = false;
                encode_params.next = &jpeg_encode_params;
            }
            int device_id = NVIMGCDCS_DEVICE_CURRENT;
            nvimgcdcsImageInfo_t out_image_info(image_info);
            fill_encode_params(params, &encode_params, &out_image_info, &device_id);

            nvimgcdcsCodeStream_t output_code_stream;
            nvimgcdcsCodeStreamCreateToFile(instance, &output_code_stream, file_name, codec_name.c_str(), &out_image_info);

            nvimgcdcsEncoder_t encoder;
            nvimgcdcsEncoderCreate(instance, &encoder, device_id);

            nvimgcdcsFuture_t future;
            nvimgcdcsEncoderEncode(encoder, &image, &output_code_stream, 1, &encode_params, &future);
            nvimgcdcsProcessingStatus_t encode_status;
            size_t status_size;
            nvimgcdcsFutureGetProcessingStatus(future, &encode_status, &status_size);
            if (encode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
                NVIMGCDCS_LOG_ERROR("Something went wrong during encoding");
                ret = NVIMGCDCS_STATUS_EXECUTION_FAILED;
            }
            nvimgcdcsFutureDestroy(future);

            nvimgcdcsEncoderDestroy(encoder);
            nvimgcdcsCodeStreamDestroy(output_code_stream);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDebugMessengerCreate(
    nvimgcdcsInstance_t instance, nvimgcdcsDebugMessenger_t* dbgMessenger, const nvimgcdcsDebugMessengerDesc_t* messengerDesc)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            if (messengerDesc == NULL) {
                messengerDesc = instance->director_.debug_messenger_.getDesc();
            }
            *dbgMessenger = new nvimgcdcsDebugMessenger(messengerDesc);
            (*dbgMessenger)->instance_ = instance;
            Logger::get().registerDebugMessenger(&(*dbgMessenger)->debug_messenger_);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDebugMessengerDestroy(nvimgcdcsDebugMessenger_t dbgMessenger)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(dbgMessenger)
            Logger::get().unregisterDebugMessenger(&dbgMessenger->debug_messenger_);
            delete dbgMessenger;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsFutureWaitForAll(nvimgcdcsFuture_t future)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(future)
            future->handle_->waitForAll();
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsFutureDestroy(nvimgcdcsFuture_t future)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(future)
            delete future;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsFutureGetProcessingStatus(
    nvimgcdcsFuture_t future,  nvimgcdcsProcessingStatus_t* processing_status, size_t* size)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(future)
            CHECK_NULL(size)
            std::vector<ProcessingResult> results (std::move(future->handle_->getAllCopy()));
            *size = results.size();
            if (processing_status) {
                auto ptr = processing_status;
                for (auto r : results) {
                    *ptr = r.status_;
                    ptr++;
                }
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}
