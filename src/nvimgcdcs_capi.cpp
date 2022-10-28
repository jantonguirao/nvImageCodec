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

#include "code_stream.h"
#include "codec.h"
#include "codec_registry.h"
#include "decode_state.h"
#include "encode_state.h"
#include "exception.h"
#include "image.h"
#include "image_decoder.h"
#include "image_encoder.h"
#include "plugin_framework.h"
#include <cstring>

#include <iostream>
#include <stdexcept>
#include <string>

using namespace nvimgcdcs;

__inline__ nvimgcdcsStatus_t getCAPICode(StatusNVIMGCDCS status)
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
        code = NVIMGCDCS_STATUS_CODESTREAM_NOT_SUPPORTED;
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
/*TEMP!!!*/ #define VERBOSE_ERRORS
#else
    #define VERBOSE_ERRORS
#endif

#define NVIMGCDCSAPI_TRY try

#ifndef VERBOSE_ERRORS
    #define NVIMGCDCSAPI_CATCH(a)                                  \
        catch (const ExceptionNVIMGCDCS& e)                        \
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
        catch (const ExceptionNVIMGCDCS& e)                           \
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
        }                                                             \
        catch (...)                                                   \
        {                                                             \
            std::cerr << "Unknown NVIMGCODECS error" << std::endl;    \
            a = NVIMGCDCS_STATUS_INTERNAL_ERROR;                      \
        }
#endif

    struct nvimgcdcsHandle
{
    nvimgcdcsHandle(
        nvimgcdcsDeviceAllocator_t* device_allocator, nvimgcdcsPinnedAllocator_t* pinned_allocator)
        : device_allocator_(device_allocator)
        , pinned_allocator_(pinned_allocator)
        , codec_registry_()
        , plugin_framework_(&codec_registry_)
    {
    }

    ~nvimgcdcsHandle() {}

    nvimgcdcsDeviceAllocator_t* device_allocator_;
    nvimgcdcsPinnedAllocator_t* pinned_allocator_;
    CodecRegistry codec_registry_;
    PluginFramework plugin_framework_;
};

struct nvimgcdcsDecoder
{
    std::unique_ptr<ImageDecoder> image_decoder_;
};

struct nvimgcdcsDecodeState
{
    std::unique_ptr<DecodeState> decode_state_;
};

struct nvimgcdcsEncoder
{
    std::unique_ptr<ImageEncoder> image_encoder_;
};

struct nvimgcdcsEncodeState
{
    std::unique_ptr<EncodeState> encode_state_;
};

struct nvimgcdcsImage
{

    explicit nvimgcdcsImage(nvimgcdcsImageInfo_t* image_info)
        : image_(image_info)
        , dev_image_buffer_(nullptr)
        , dev_image_buffer_size_(0)
    {
    }

    ~nvimgcdcsImage()
    {
        if (dev_image_buffer_) {
            CHECK_CUDA(cudaFree(dev_image_buffer_));
        }
    }
    nvimgcdcsInstance_t nvimgcdcs_instance_;
    Image image_;
    unsigned char* dev_image_buffer_;
    size_t dev_image_buffer_size_;
};

nvimgcdcsStatus_t nvimgcdcsInstanceCreate(
    nvimgcdcsInstance_t* instance, nvimgcdcsInstanceCreateInfo_t createInfo)
{
    nvimgcdcsStatus_t ret         = NVIMGCDCS_STATUS_SUCCESS;
    nvimgcdcsInstance_t nvimgcdcs = nullptr;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance);
            nvimgcdcs =
                new nvimgcdcsHandle(createInfo.device_allocator, createInfo.pinned_allocator);
            nvimgcdcs->plugin_framework_.discoverAndLoadExtModules();
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

struct nvimgcdcsCodeStream
{
    explicit nvimgcdcsCodeStream(CodecRegistry* codec_registry)
        : code_stream_(codec_registry)
    {
    }
    nvimgcdcs::CodeStream code_stream_;
};

static nvimgcdcsStatus_t nvimgcdcsStreamCreate(
    nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle)
{
    nvimgcdcsStatus_t ret        = NVIMGCDCS_STATUS_SUCCESS;
    nvimgcdcsCodeStream_t stream = nullptr;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle);
            stream         = new nvimgcdcsCodeStream(&instance->codec_registry_);
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

nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromHostMem(nvimgcdcsInstance_t instance,
    nvimgcdcsCodeStream_t* stream_handle, unsigned char* data, size_t size)
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

nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToFile(nvimgcdcsInstance_t instance,
    nvimgcdcsCodeStream_t* stream_handle, const char* file_name, const char* codec_name)
{
    nvimgcdcsStatus_t ret = nvimgcdcsStreamCreate(instance, stream_handle);
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle)
            CHECK_NULL(file_name)
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                (*stream_handle)->code_stream_.setOutputToFile(file_name, codec_name);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToHostMem(nvimgcdcsInstance_t instance,
    nvimgcdcsCodeStream_t* stream_handle, unsigned char* output_buffer, size_t length,
    const char* codec_name)
{
    nvimgcdcsStatus_t ret = nvimgcdcsStreamCreate(instance, stream_handle);
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle)
            CHECK_NULL(output_buffer)
            CHECK_NULL(length)
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                (*stream_handle)->code_stream_.setOutputToHostMem(output_buffer, length, codec_name);
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

nvimgcdcsStatus_t nvimgcdcsCodeStreamGetImageInfo(
    nvimgcdcsCodeStream_t stream_handle, nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(stream_handle)
            CHECK_NULL(image_info)
            stream_handle->code_stream_.getImageInfo(image_info);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamSetImageInfo(
    nvimgcdcsCodeStream_t stream_handle, nvimgcdcsImageInfo_t* image_info)
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

nvimgcdcsStatus_t nvimgcdcsCodeStreamGetCodecName(
    nvimgcdcsCodeStream_t stream_handle, char* codec_name)
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
            strcpy(codec_name,codec_name_.c_str());
            #endif
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderCreate(nvimgcdcsInstance_t instance, nvimgcdcsDecoder_t* decoder,
    nvimgcdcsCodeStream_t stream, nvimgcdcsDecodeParams_t* params)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(decoder)
            CHECK_NULL(stream)
            Codec* codec = stream->code_stream_.getCodec();
            CHECK_NULL(codec)
            std::unique_ptr<ImageDecoder> image_decoder =
                codec->createDecoder(stream->code_stream_.getCodeStreamDesc(), params);
            if (image_decoder) {
                *decoder                   = new nvimgcdcsDecoder();
                (*decoder)->image_decoder_ = std::move(image_decoder);
            } else {
                ret = NVIMGCDCS_STATUS_CODESTREAM_NOT_SUPPORTED;
            }
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

nvimgcdcsStatus_t nvimgcdcsDecoderDecode(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStream_t stream,
    nvimgcdcsImage_t image, nvimgcdcsDecodeParams_t* params)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(stream)
            CHECK_NULL(image)
            CHECK_NULL(params)

            decoder->image_decoder_->decode(&stream->code_stream_, &image->image_, params);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderDecodeBatch(nvimgcdcsDecoder_t decoder,
    nvimgcdcsDecodeParams_t* params, nvimgcdcsContainer_t container, int batchSize,
    nvimgcdcsImage_t* image)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            assert(!"TODO");
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderGetDecodedImage(nvimgcdcsDecoder_t decoder, bool blocking,
    nvimgcdcsImage_t* image, nvimgcdcsDecodeStatus_t* decode_status)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            assert(!"TODO");
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderGetCapabilities(
    nvimgcdcsDecoder_t decoder, nvimgcdcsCapability_t* decoder_capabilites, size_t* size)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            assert(!"TODO");
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderCanUseDecodeState(
    nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t decodeState, bool* canUse)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            assert(!"TODO");
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecodeStateCreate(
    nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(decode_state)
            std::unique_ptr<DecodeState> decode_state_ =
                decoder->image_decoder_->createDecodeState();
            if (decode_state_) {
                *decode_state                  = new nvimgcdcsDecodeState();
                (*decode_state)->decode_state_ = std::move(decode_state_);
            } else {
                ret = NVIMGCDCS_STATUS_INTERNAL_ERROR;
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}
 
nvimgcdcsStatus_t nvimgcdcsDecodeStateDestroy(nvimgcdcsDecodeState_t decode_state)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decode_state)
            delete decode_state;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsImageCreate(
    nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image, nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)

            *image                        = new nvimgcdcsImage(image_info);
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

nvimgcdcsStatus_t nvimgcdcsImageSetHostBuffer(nvimgcdcsImage_t image, void* buffer, size_t size)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            image->image_.setHostBuffer(buffer, size);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsImageGetHostBuffer(nvimgcdcsImage_t image, void** buffer, size_t* size)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            image->image_.getHostBuffer(buffer, size);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}
nvimgcdcsStatus_t nvimgcdcsImageSetDeviceBuffer(nvimgcdcsImage_t image, void* buffer, size_t size)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            image->image_.setDeviceBuffer(buffer, size);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}
nvimgcdcsStatus_t nvimgcdcsImageGetDeviceBuffer(nvimgcdcsImage_t image, void** buffer, size_t* size)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            image->image_.getDeviceBuffer(buffer, size);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}
nvimgcdcsStatus_t nvimgcdcsImageSetImageInfo(
    nvimgcdcsImage_t image, nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            image->image_.setImageInfo(image_info);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}
nvimgcdcsStatus_t nvimgcdcsImageGetImageInfo(
    nvimgcdcsImage_t image, nvimgcdcsImageInfo_t* image_info)
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

nvimgcdcsStatus_t nvimgcdcsImageAttachDecodeState(
    nvimgcdcsImage_t image, nvimgcdcsDecodeState_t decode_state)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            CHECK_NULL(decode_state)
            image->image_.attachDecodeState(decode_state->decode_state_.get());
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsImageDetachDecodeState(nvimgcdcsImage_t image)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            image->image_.detachDecodeState();
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsImageAttachEncodeState(
    nvimgcdcsImage_t image, nvimgcdcsEncodeState_t encode_state)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            CHECK_NULL(encode_state)
            image->image_.attachEncodeState(encode_state->encode_state_.get());
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsImageDetachEncodeState(nvimgcdcsImage_t image)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            image->image_.detachEncodeState();
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncoderCreate(nvimgcdcsInstance_t instance, nvimgcdcsEncoder_t* encoder,
    nvimgcdcsCodeStream_t stream, nvimgcdcsEncodeParams_t* params)

{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(encoder)
            CHECK_NULL(stream)
            Codec* codec = stream->code_stream_.getCodec();
            CHECK_NULL(codec)
            std::unique_ptr<ImageEncoder> image_encoder =
                codec->createEncoder(stream->code_stream_.getCodeStreamDesc(), params);
            if (image_encoder) {
                *encoder                   = new nvimgcdcsEncoder();
                (*encoder)->image_encoder_ = std::move(image_encoder);
            } else {
                ret = NVIMGCDCS_STATUS_CODESTREAM_NOT_SUPPORTED;
            }
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

nvimgcdcsStatus_t nvimgcdcsEncoderEncode(nvimgcdcsEncoder_t encoder, nvimgcdcsCodeStream_t stream,
    nvimgcdcsImage_t image, nvimgcdcsEncodeParams_t* params)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(stream)
            CHECK_NULL(image)
            CHECK_NULL(params)

            encoder->image_encoder_->encode(&stream->code_stream_, &image->image_, params);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncodeStateCreate(
    nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t* encode_state)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(encode_state)
            std::unique_ptr<EncodeState> encode_state_ =
                encoder->image_encoder_->createEncodeState();
            if (encode_state_) {
                *encode_state                  = new nvimgcdcsEncodeState();
                (*encode_state)->encode_state_ = std::move(encode_state_);
            } else {
                ret = NVIMGCDCS_STATUS_INTERNAL_ERROR;
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncodeStateDestroy(nvimgcdcsEncodeState_t encode_state)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encode_state)
            delete encode_state;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsImgRead(nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image, const char* file_name)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
    {
        CHECK_NULL(instance)
        CHECK_NULL(image) 
        CHECK_NULL(file_name)

        nvimgcdcsCodeStream_t code_stream;
        nvimgcdcsCodeStreamCreateFromFile(instance, &code_stream, file_name);
        nvimgcdcsImageInfo_t image_info;
        nvimgcdcsCodeStreamGetImageInfo(code_stream, &image_info);
        char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
        nvimgcdcsCodeStreamGetCodecName(code_stream, codec_name);
        int bytes_per_element = image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 1 : 2;
        image_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        nvimgcdcsDecodeParams_t decode_params;
        decode_params.backend.useGPU = true;

        nvimgcdcsDecoder_t decoder;
        nvimgcdcsDecoderCreate(instance, &decoder, code_stream, &decode_params);

        nvimgcdcsDecodeState_t decode_state;
        nvimgcdcsDecodeStateCreate(decoder, &decode_state);

        unsigned char* image_buffer;
        CHECK_CUDA(
            cudaMallocPitch((void**)&image_buffer, &image_info.component_info[0].pitch_in_bytes,
                image_info.image_width * bytes_per_element,
                image_info.image_height * image_info.num_components));
        image_info.component_info[1].pitch_in_bytes = image_info.component_info[0].pitch_in_bytes;
        image_info.component_info[2].pitch_in_bytes = image_info.component_info[0].pitch_in_bytes;
        size_t image_buffer_size                    = image_info.component_info[0].pitch_in_bytes *
                                   image_info.image_height * image_info.num_components;

        nvimgcdcsImageCreate(instance, image, &image_info);
        (*image)->dev_image_buffer_ = image_buffer;
        (*image)->dev_image_buffer_size_ = image_buffer_size;
        nvimgcdcsImageSetDeviceBuffer(*image, image_buffer, image_buffer_size);
        nvimgcdcsImageAttachDecodeState(*image, decode_state);
        nvimgcdcsDecoderDecode(decoder, code_stream, *image, &decode_params);
        cudaDeviceSynchronize();
        nvimgcdcsImageDetachDecodeState(*image);
        nvimgcdcsDecodeStateDestroy(decode_state);
        nvimgcdcsDecoderDestroy(decoder);
        nvimgcdcsCodeStreamDestroy(code_stream);
    } NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsImgWrite(
    nvimgcdcsInstance_t instance, nvimgcdcsImage_t image, const char* file_name, const int* params)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(image)
            CHECK_NULL(file_name)

            nvimgcdcsImageInfo_t image_info;
            nvimgcdcsImageGetImageInfo(image, &image_info);

            nvimgcdcsCodeStream_t bmp_code_stream;
            nvimgcdcsCodeStreamCreateToFile(instance, &bmp_code_stream, file_name, "bmp" /*TODO*/);
            nvimgcdcsCodeStreamSetImageInfo(bmp_code_stream, &image_info);

            nvimgcdcsEncodeParams_t encode_params;
            encode_params.backend.useCPU = true;
            encode_params.target_psnr    = 50; //TODO
            encode_params.codec          = "bmp" /*TODO*/;

            nvimgcdcsEncoder_t encoder;
            nvimgcdcsEncoderCreate(instance, &encoder, bmp_code_stream, &encode_params);
            nvimgcdcsEncodeState_t encode_state;
            nvimgcdcsEncodeStateCreate(encoder, &encode_state);
            nvimgcdcsImageAttachEncodeState(image, encode_state);
            nvimgcdcsEncoderEncode(encoder, bmp_code_stream, image, &encode_params);
            cudaDeviceSynchronize();
            nvimgcdcsImageDetachEncodeState(image);
            nvimgcdcsEncodeStateDestroy(encode_state);
            nvimgcdcsEncoderDestroy(encoder);
            nvimgcdcsCodeStreamDestroy(bmp_code_stream);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

