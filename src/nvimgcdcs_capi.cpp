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
        catch (const Exception& e)                           \
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

struct nvimgcdcsDecodeState
{
    std::unique_ptr<IDecodeState> decode_state_;
};

struct nvimgcdcsEncoder
{
    nvimgcdcsInstance_t instance_;
    std::unique_ptr<IImageEncoder> image_encoder_;
};

struct nvimgcdcsEncodeState
{
    std::unique_ptr<IEncodeState> encode_state_;
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
    unsigned char* dev_image_buffer_;
    size_t dev_image_buffer_size_;
    std::vector<unsigned char> host_image_buffer_;
};

nvimgcdcsStatus_t nvimgcdcsInstanceCreate(
    nvimgcdcsInstance_t* instance, nvimgcdcsInstanceCreateInfo_t create_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    nvimgcdcsInstance_t nvimgcdcs = nullptr;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance);
            nvimgcdcs = new nvimgcdcsInstance(create_info);
            if (create_info.load_extension_modules) {
                nvimgcdcs->director_.plugin_framework_.discoverAndLoadExtModules();
            }

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

nvimgcdcsStatus_t nvimgcdcsExtensionCreate(nvimgcdcsInstance_t instance,
    nvimgcdcsExtension_t* extension, nvimgcdcsExtensionDesc_t* extension_desc)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(extension_desc)
            nvimgcdcsExtension_t extension_ext_handle;
            ret = instance->director_.plugin_framework_.registerExtension(
                &extension_ext_handle, extension_desc);
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

            return extension->nvimgcdcs_instance_->director_.plugin_framework_.unregisterExtension(
                extension->extension_ext_handle_);
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

static nvimgcdcsStatus_t nvimgcdcsStreamCreate(
    nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* stream_handle)
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
                (*stream_handle)
                    ->code_stream_.setOutputToHostMem(output_buffer, length, codec_name);
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
            strcpy(codec_name, codec_name_.c_str());
#endif
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderGenericCreate(
    nvimgcdcsInstance_t instance, nvimgcdcsDecoder_t* decoder)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(decoder)
            std::unique_ptr<IImageDecoder> image_decoder =
                instance->director_.createGenericDecoder();
            *decoder = new nvimgcdcsDecoder();
            (*decoder)->image_decoder_ = std::move(image_decoder);
            (*decoder)->instance_ = instance;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderCreate(nvimgcdcsInstance_t instance, nvimgcdcsDecoder_t* decoder,
    nvimgcdcsCodeStream_t stream, nvimgcdcsImage_t image, const nvimgcdcsDecodeParams_t* params)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(decoder)
            CHECK_NULL(stream)
            CHECK_NULL(image)
            ICodec* codec = stream->code_stream_.getCodec();
            CHECK_NULL(codec)
            std::unique_ptr<IImageDecoder> image_decoder = codec->createDecoder(
                stream->code_stream_.getCodeStreamDesc(), image->image_.getImageDesc(), params);
            if (image_decoder) {
                *decoder = new nvimgcdcsDecoder();
                (*decoder)->image_decoder_ = std::move(image_decoder);
                (*decoder)->instance_ = instance;
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

nvimgcdcsStatus_t nvimgcdcsDecoderCanDecode(nvimgcdcsDecoder_t decoder, bool* result,
    nvimgcdcsCodeStream_t stream, nvimgcdcsImage_t image, const nvimgcdcsDecodeParams_t* params)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(result)
            CHECK_NULL(stream)
            CHECK_NULL(image)
            CHECK_NULL(params)
            *result = decoder->image_decoder_->canDecode(
                stream->code_stream_.getCodeStreamDesc(), image->image_.getImageDesc(), params);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderDecode(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStream_t stream,
    nvimgcdcsImage_t image, const nvimgcdcsDecodeParams_t* params, nvimgcdcsFuture_t* future,
    bool blocking)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(stream)
            CHECK_NULL(image)
            CHECK_NULL(params)
            auto int_future = decoder->image_decoder_->decode(&stream->code_stream_, &image->image_, params);
            if (blocking) {
                int_future->waitForAll();
            } else {
                CHECK_NULL(future)
                *future = new nvimgcdcsFuture();
                (*future)->handle_ = std::move(int_future);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderDecodeBatch(nvimgcdcsDecoder_t decoder,
    nvimgcdcsDecodeState_t decode_state_batch, nvimgcdcsCodeStream_t* streams,
    nvimgcdcsImage_t* images, int batch_size, nvimgcdcsDecodeParams_t* params,
    nvimgcdcsFuture_t* future, bool blocking)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(decode_state_batch)
            CHECK_NULL(streams)
            CHECK_NULL(images)
            CHECK_NULL(params)

            std::vector<nvimgcdcs::ICodeStream*> internal_code_streams;
            std::vector<nvimgcdcs::IImage*> internal_images;

            for (int i = 0; i < batch_size; ++i) {
                internal_code_streams.push_back(&streams[i]->code_stream_);
                internal_images.push_back(&images[i]->image_);
            }
            auto int_future =decoder->image_decoder_->decodeBatch(decode_state_batch->decode_state_.get(),
                internal_code_streams, internal_images, params);

            if (blocking) {
                int_future->waitForAll();
            } else {
                CHECK_NULL(future)
                *future = new nvimgcdcsFuture();
                (*future)->handle_ = std::move(int_future);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderGetCapabilities(
    nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(size)
            decoder->image_decoder_->getCapabilities(capabilities, size);
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
    nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state, cudaStream_t cuda_stream)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(decode_state)
            std::unique_ptr<IDecodeState> decode_state_ =
                decoder->image_decoder_->createDecodeState(cuda_stream);
            if (decode_state_) {
                *decode_state = new nvimgcdcsDecodeState();
                (*decode_state)->decode_state_ = std::move(decode_state_);
            } else {
                ret = NVIMGCDCS_STATUS_INTERNAL_ERROR;
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecodeStateBatchCreate(
    nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state, cudaStream_t cuda_stream)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(decode_state)

            std::unique_ptr<IDecodeState> decode_state_ =
                decoder->image_decoder_->createDecodeStateBatch(cuda_stream);
            if (decode_state_) {
                *decode_state = new nvimgcdcsDecodeState();
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

nvimgcdcsStatus_t nvimgcdcsImageCreate(nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            CHECK_NULL(instance)

            *image = new nvimgcdcsImage();
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

nvimgcdcsStatus_t nvimgcdcsImageGetProcessingStatus(
    nvimgcdcsImage_t image, nvimgcdcsProcessingStatus_t* processing_status)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(image)
            CHECK_NULL(processing_status)
            *processing_status = image->image_.getProcessingStatus();
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}


NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderGenericCreate(
    nvimgcdcsInstance_t instance, nvimgcdcsEncoder_t* encoder)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(encoder)
            std::unique_ptr<IImageEncoder> image_encoder =
                instance->director_.createGenericEncoder();
            *encoder = new nvimgcdcsEncoder();
            (*encoder)->image_encoder_ = std::move(image_encoder);
            (*encoder)->instance_ = instance;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncoderCreate(nvimgcdcsInstance_t instance, nvimgcdcsEncoder_t* encoder,
    nvimgcdcsImage_t image, nvimgcdcsCodeStream_t stream, const nvimgcdcsEncodeParams_t* params)

{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(encoder)
            CHECK_NULL(image)
            CHECK_NULL(stream)
            ICodec* codec = stream->code_stream_.getCodec();
            CHECK_NULL(codec)
            std::unique_ptr<IImageEncoder> image_encoder = codec->createEncoder(
                image->image_.getImageDesc(), stream->code_stream_.getCodeStreamDesc(), params);
            if (image_encoder) {
                *encoder = new nvimgcdcsEncoder();
                (*encoder)->image_encoder_ = std::move(image_encoder);
                (*encoder)->instance_ = instance;
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

nvimgcdcsStatus_t nvimgcdcsEncoderCanEncode(nvimgcdcsEncoder_t encoder, bool* result,
    nvimgcdcsImage_t image, nvimgcdcsCodeStream_t stream, const nvimgcdcsEncodeParams_t* params)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(result)
            CHECK_NULL(image)
            CHECK_NULL(stream)
            *result = encoder->image_encoder_->canEncode(
                image->image_.getImageDesc(), stream->code_stream_.getCodeStreamDesc(), params);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncoderEncode(nvimgcdcsEncoder_t encoder, nvimgcdcsCodeStream_t stream,
    nvimgcdcsImage_t image, const nvimgcdcsEncodeParams_t* params, nvimgcdcsFuture_t* future,
    bool blocking)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(stream)
            CHECK_NULL(image)
            CHECK_NULL(params)
            auto int_future = encoder->image_encoder_->encode(&stream->code_stream_, &image->image_, params);
            if (blocking) {
                int_future->waitForAll();
            } else {
                CHECK_NULL(future)
                *future = new nvimgcdcsFuture();
                (*future)->handle_ = std::move(int_future);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncoderEncodeBatch(nvimgcdcsEncoder_t encoder,
    nvimgcdcsEncodeState_t encode_state_batch, nvimgcdcsImage_t* images,
    nvimgcdcsCodeStream_t* streams, int batch_size, nvimgcdcsEncodeParams_t* params,
    nvimgcdcsFuture_t* future, bool blocking)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(encode_state_batch)
            CHECK_NULL(streams)
            CHECK_NULL(images)
            CHECK_NULL(params)

            std::vector<nvimgcdcs::ICodeStream*> internal_code_streams;
            std::vector<nvimgcdcs::IImage*> internal_images;

            for (int i = 0; i < batch_size; ++i) {
                internal_code_streams.push_back(&streams[i]->code_stream_);
                internal_images.push_back(&images[i]->image_);
            }

            auto int_future = encoder->image_encoder_->encodeBatch(encode_state_batch->encode_state_.get(),
                internal_images, internal_code_streams, params);
            if (blocking) {
                int_future->waitForAll();
            } else {
                CHECK_NULL(future)
                *future = new nvimgcdcsFuture();
                (*future)->handle_ = std::move(int_future);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncoderGetCapabilities(
    nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(size)
            encoder->image_encoder_->getCapabilities(capabilities, size);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncodeStateCreate(
    nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t* encode_state, cudaStream_t cuda_stream)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(encode_state)
            std::unique_ptr<IEncodeState> encode_state_ =
                encoder->image_encoder_->createEncodeState(cuda_stream);
            if (encode_state_) {
                *encode_state = new nvimgcdcsEncodeState();
                (*encode_state)->encode_state_ = std::move(encode_state_);
            } else {
                ret = NVIMGCDCS_STATUS_INTERNAL_ERROR;
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncodeStateBatchCreate(nvimgcdcsEncoder_t encoder,
    nvimgcdcsEncodeState_t* encode_state_batch, cudaStream_t cuda_stream)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
           CHECK_NULL(encoder)
           CHECK_NULL(encode_state_batch)


           std::unique_ptr<IEncodeState> encode_state_ =
               encoder->image_encoder_->createEncodeStateBatch(cuda_stream);

           if (encode_state_) {
                *encode_state_batch = new nvimgcdcsEncodeState();
                (*encode_state_batch)->encode_state_ = std::move(encode_state_);
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

nvimgcdcsStatus_t nvimgcdcsImRead(
    nvimgcdcsInstance_t instance, nvimgcdcsImage_t* image, const char* file_name, int flags)
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
            int bytes_per_element =
                image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 1 : 2;
            nvimgcdcsDecodeParams_t decode_params;
            memset(&decode_params, 0, sizeof(nvimgcdcsDecodeParams_t));
            decode_params.enable_color_conversion = flags & NVIMGCDCS_IMREAD_COLOR;
            decode_params.enable_orientation = !(flags & NVIMGCDCS_IMREAD_IGNORE_ORIENTATION);
            if (decode_params.enable_orientation) {
                decode_params.orientation.rotated =
                    image_info.orientation.rotated == 90
                        ? 270
                        : (image_info.orientation.rotated == 270 ? 90 : 0);
                if (decode_params.orientation.rotated) {
                    auto tmp = image_info.image_width;
                    image_info.image_width = image_info.image_height;
                    image_info.image_height = tmp;
                }
            }
            nvimgcdcsImageCreate(instance, image);
            //TODO define image_info for requested output

            nvimgcdcsDecoder_t decoder;
            nvimgcdcsDecoderCreate(instance, &decoder, code_stream, *image, &decode_params);

            nvimgcdcsDecodeState_t decode_state;
            nvimgcdcsDecodeStateCreate(decoder, &decode_state, nullptr);

            size_t capabilities_size;
            nvimgcdcsDecoderGetCapabilities(decoder, nullptr, &capabilities_size);
            const nvimgcdcsCapability_t* capabilities_ptr;
            nvimgcdcsDecoderGetCapabilities(decoder, &capabilities_ptr, &capabilities_size);

            bool is_host_output =
                std::find(capabilities_ptr,
                    capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                    NVIMGCDCS_CAPABILITY_HOST_OUTPUT) !=
                capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
            bool is_device_output =
                std::find(capabilities_ptr,
                    capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                    NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT) !=
                capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);

            bool is_interleaved = static_cast<int>(image_info.sample_format) % 2 == 0;
            unsigned char* device_buffer;
            if (is_device_output) {
                size_t device_pitch_in_bytes = 0;
                CHECK_CUDA(cudaMallocPitch((void**)&device_buffer, &device_pitch_in_bytes,
                    image_info.image_width * bytes_per_element *
                        (is_interleaved ? image_info.num_components : 1),
                    image_info.image_height * (is_interleaved ? 1 : image_info.num_components)));
                image_info.component_info[0].device_pitch_in_bytes = device_pitch_in_bytes;
                image_info.component_info[1].device_pitch_in_bytes = device_pitch_in_bytes;
                image_info.component_info[2].device_pitch_in_bytes = device_pitch_in_bytes;

                size_t device_buffer_size =
                    device_pitch_in_bytes * image_info.image_height * image_info.num_components;
                (*image)->dev_image_buffer_ = device_buffer;
                (*image)->dev_image_buffer_size_ = device_buffer_size;
                nvimgcdcsImageSetDeviceBuffer(*image, device_buffer, device_buffer_size);
            }

            if (is_host_output) {
                (*image)->host_image_buffer_.resize(
                    image_info.image_width * image_info.image_height * image_info.num_components);
                nvimgcdcsImageSetHostBuffer(*image, (*image)->host_image_buffer_.data(),
                    (*image)->host_image_buffer_.size());
                image_info.component_info[0].host_pitch_in_bytes =
                    image_info.image_width * (is_interleaved ? image_info.num_components : 1);
                image_info.component_info[1].host_pitch_in_bytes =
                    image_info.image_width * (is_interleaved ? image_info.num_components : 1);
                image_info.component_info[2].host_pitch_in_bytes =
                    image_info.image_width * (is_interleaved ? image_info.num_components : 1);
            }

            image_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
            image_info.color_space = NVIMGCDCS_COLORSPACE_SRGB;

            nvimgcdcsImageSetImageInfo(*image, &image_info);
            nvimgcdcsImageAttachDecodeState(*image, decode_state);
            nvimgcdcsDecoderDecode(decoder, code_stream, *image, &decode_params, nullptr, true);
            nvimgcdcsProcessingStatus_t decode_status;
            nvimgcdcsImageGetProcessingStatus(*image, &decode_status);
            if (decode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
                NVIMGCDCS_LOG_ERROR("Something went wrong during decoding");
            }
            nvimgcdcsImageDetachDecodeState(*image);
            nvimgcdcsDecodeStateDestroy(decode_state);
            nvimgcdcsDecoderDestroy(decoder);
            nvimgcdcsCodeStreamDestroy(code_stream);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

static std::map<std::string, std::string> ext2codec = {{".bmp", "bmp"}, {".j2c", "jpeg2k"},
    {".j2k", "jpeg2k"}, {".jp2", "jpeg2k"}, {".tiff", "tiff"}, {".tif", "tiff"}, {".jpg", "jpeg"},
    {".jpeg", "jpeg"}, {".ppm", "pxm"}, {".pgm", "pxm"}, {".pbm", "pxm"}};

inline size_t sample_type_to_bytes_per_element(nvimgcdcsSampleDataType_t sample_type)
{
    //TODO for more types
    return sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 1 : 2;
}

static void fill_encode_params(
    const int* params, nvimgcdcsEncodeParams_t* encode_params, nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsJpegEncodeParams_t* jpeg_encode_params =
        static_cast<nvimgcdcsJpegEncodeParams_t*>(encode_params->next);
    nvimgcdcsJpeg2kEncodeParams_t* jpeg2k_encode_params =
        static_cast<nvimgcdcsJpeg2kEncodeParams_t*>(encode_params->next);

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
            nvimgcdcsImwriteSamplingFactor_t sampling_factor =
                static_cast<nvimgcdcsImwriteSamplingFactor_t>(*param);
            std::map<nvimgcdcsImwriteSamplingFactor_t, nvimgcdcsChromaSubsampling_t> sf2css = {
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
                image_info->sampling = it->second;
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
        case NVIMGCDCS_IMWRITE_MCT_MODE: {
            param++;
            encode_params->mct_mode = static_cast<nvimgcdcsMctMode_t>(*param);
            break;
        }
        default:
            break;
        };
        param++;
    }
}

nvimgcdcsStatus_t nvimgcdcsImWrite(
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
            fs::path file_path(file_name);

            std::string codec_name = "bmp";
            if (file_path.has_extension()) {
                std::string extension = file_path.extension().string();
                auto it = ext2codec.find(extension);
                if (it != ext2codec.end()) {
                    codec_name = it->second;
                }
            }

            if (image_info.sampling == NVIMGCDCS_SAMPLING_UNKNOWN ||
                image_info.sampling == NVIMGCDCS_SAMPLING_NOT_SUPPORTED)
                image_info.sampling = NVIMGCDCS_SAMPLING_444;
            nvimgcdcsEncodeParams_t encode_params;
            memset(&encode_params, 0, sizeof(nvimgcdcsEncodeParams_t));
            //Defaults
            encode_params.quality = 95;
            encode_params.target_psnr = 50;
            encode_params.mct_mode = NVIMGCDCS_MCT_MODE_RGB;

            nvimgcdcsJpeg2kEncodeParams_t jpeg2k_encode_params;
            nvimgcdcsJpegEncodeParams_t jpeg_encode_params;
            if (codec_name == "jpeg2k") {
                memset(&jpeg2k_encode_params, 0, sizeof(jpeg2k_encode_params));
                jpeg2k_encode_params.type = NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
                jpeg2k_encode_params.stream_type = file_path.extension().string() == ".jp2"
                                                       ? NVIMGCDCS_JPEG2K_STREAM_JP2
                                                       : NVIMGCDCS_JPEG2K_STREAM_J2K;
                jpeg2k_encode_params.prog_order =
                    NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL; //TODO Support for all j2k progression orders
                jpeg2k_encode_params.num_layers = 1;
                jpeg2k_encode_params.irreversible = 1;
                //jpeg2k_encode_params.rsiz = ;
                jpeg2k_encode_params.enable_SOP_marker = 0;
                jpeg2k_encode_params.enable_EPH_marker = 0;
                jpeg2k_encode_params.num_resolutions = 5;
                jpeg2k_encode_params.code_block_w = 64;
                jpeg2k_encode_params.code_block_h = 64;
                // jpeg2k_encode_params.encode_modes;
                jpeg2k_encode_params.enable_custom_precincts = 0;

                encode_params.next = &jpeg2k_encode_params;
            } else if (codec_name == "jpeg") {
                memset(&jpeg_encode_params, 0, sizeof(jpeg_encode_params));
                jpeg_encode_params.type = NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS;
                jpeg_encode_params.encoding = NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT;
                jpeg_encode_params.optimized_huffman = false;
                encode_params.next = &jpeg_encode_params;
            }

            nvimgcdcsImageInfo_t out_image_info(image_info);
            fill_encode_params(params, &encode_params, &out_image_info);

            nvimgcdcsCodeStream_t output_code_stream;
            nvimgcdcsCodeStreamCreateToFile(
                instance, &output_code_stream, file_name, codec_name.c_str());
            nvimgcdcsCodeStreamSetImageInfo(output_code_stream, &out_image_info);
            nvimgcdcsImageSetImageInfo(image, &image_info);

            nvimgcdcsEncoder_t encoder;
            nvimgcdcsEncoderCreate(instance, &encoder, image, output_code_stream, &encode_params);
            nvimgcdcsEncodeState_t encode_state;
            nvimgcdcsEncodeStateCreate(encoder, &encode_state, nullptr);
            nvimgcdcsImageAttachEncodeState(image, encode_state);

            size_t capabilities_size;
            nvimgcdcsEncoderGetCapabilities(encoder, nullptr, &capabilities_size);
            const nvimgcdcsCapability_t* capabilities_ptr;
            nvimgcdcsEncoderGetCapabilities(encoder, &capabilities_ptr, &capabilities_size);
#if 0
            std::span<const nvimgcdcsCapability_t> encoder_capabilties{
                capabilities_ptr, capabilities_size};

            bool is_host_input = std::find(encoder_capabilties.begin(), encoder_capabilties.end(),
                                     NVIMGCDCS_CAPABILITY_HOST_INPUT) != encoder_capabilties.end();
            bool is_device_input =
                std::find(encoder_capabilties.begin(), encoder_capabilties.end(),
                    NVIMGCDCS_CAPABILITY_DEVICE_INPUT) != encoder_capabilties.end();
#else
            bool is_host_input =
                std::find(capabilities_ptr,
                    capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                    NVIMGCDCS_CAPABILITY_HOST_INPUT) !=
                capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
            bool is_device_input =
                std::find(capabilities_ptr,
                    capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                    NVIMGCDCS_CAPABILITY_DEVICE_INPUT) !=
                capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
#endif
            void* device_buffer = nullptr;
            size_t device_buffer_size = 0;
            nvimgcdcsImageGetDeviceBuffer(image, &device_buffer, &device_buffer_size);

            void* host_buffer = nullptr;
            size_t host_buffer_size = 0;
            nvimgcdcsImageGetHostBuffer(image, &host_buffer, &host_buffer_size);
            bool is_interleaved = static_cast<int>(image_info.sample_format) % 2 == 0;
            if (is_device_input && device_buffer == nullptr) {
                //TODO use custom allocators
                size_t bytes_per_element = sample_type_to_bytes_per_element(image_info.sample_type);
                size_t device_pitch_in_bytes = 0;
                CHECK_CUDA(cudaMallocPitch((void**)&device_buffer, &device_pitch_in_bytes,
                    image_info.image_width * bytes_per_element *
                        (is_interleaved ? image_info.num_components : 1),
                    image_info.image_height * (is_interleaved ? 1 : image_info.num_components)));
                image_info.component_info[0].device_pitch_in_bytes = device_pitch_in_bytes;
                image_info.component_info[1].device_pitch_in_bytes = device_pitch_in_bytes;
                image_info.component_info[2].device_pitch_in_bytes = device_pitch_in_bytes;
                device_buffer_size =
                    device_pitch_in_bytes * image_info.image_height * image_info.num_components;
                nvimgcdcsImageSetDeviceBuffer(image, device_buffer, device_buffer_size);
                nvimgcdcsImageSetImageInfo(image, &image_info);

                if (host_buffer) {
                    CHECK_CUDA(cudaMemcpy2D(device_buffer,
                        (size_t)image_info.component_info[0].device_pitch_in_bytes,
                        image->host_image_buffer_.data(),
                        (size_t)image_info.component_info[0].host_pitch_in_bytes,
                        image_info.image_width,
                        image_info.image_height * (is_interleaved ? 1 : image_info.num_components),
                        cudaMemcpyHostToDevice));
                } else {
                    nvimgcdcsImageDetachEncodeState(image);
                    nvimgcdcsEncodeStateDestroy(encode_state);
                    nvimgcdcsEncoderDestroy(encoder);
                    nvimgcdcsCodeStreamDestroy(output_code_stream);
                    return NVIMGCDCS_STATUS_INVALID_PARAMETER;
                }
            } else if (is_host_input && host_buffer == nullptr) {
                image->host_image_buffer_.resize(
                    image_info.image_width * image_info.image_height *
                    image_info.num_components); //TODO more bytes per sample
                image_info.component_info[0].host_pitch_in_bytes =
                    image_info.image_width * (is_interleaved ? image_info.num_components : 1);
                image_info.component_info[1].host_pitch_in_bytes =
                    image_info.image_width * (is_interleaved ? image_info.num_components : 1);
                image_info.component_info[2].host_pitch_in_bytes =
                    image_info.image_width * (is_interleaved ? image_info.num_components : 1);
                nvimgcdcsImageSetHostBuffer(
                    image, image->host_image_buffer_.data(), image->host_image_buffer_.size());
                nvimgcdcsImageSetImageInfo(image, &image_info);
                if (device_buffer) {
                    CHECK_CUDA(cudaMemcpy2D(image->host_image_buffer_.data(),
                        (size_t)image_info.component_info[0].host_pitch_in_bytes, device_buffer,
                        (size_t)image_info.component_info[0].device_pitch_in_bytes,
                        (size_t)image_info.component_info[0].host_pitch_in_bytes,
                        image_info.image_height * (is_interleaved ? 1 : image_info.num_components),
                        cudaMemcpyDeviceToHost));
                } else {
                    nvimgcdcsImageDetachEncodeState(image);
                    nvimgcdcsEncodeStateDestroy(encode_state);
                    nvimgcdcsEncoderDestroy(encoder);
                    nvimgcdcsCodeStreamDestroy(output_code_stream);
                    return NVIMGCDCS_STATUS_INVALID_PARAMETER;
                }
            }

            nvimgcdcsEncoderEncode(encoder, output_code_stream, image, &encode_params, nullptr, true);
            nvimgcdcsProcessingStatus_t encode_status;
            nvimgcdcsImageGetProcessingStatus(image, &encode_status);
            if (encode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
                NVIMGCDCS_LOG_ERROR("Something went wrong during encoding");
            }
            nvimgcdcsImageDetachEncodeState(image);
            nvimgcdcsEncodeStateDestroy(encode_state);
            nvimgcdcsEncoderDestroy(encoder);
            nvimgcdcsCodeStreamDestroy(output_code_stream);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDebugMessengerCreate(nvimgcdcsInstance_t instance,
    nvimgcdcsDebugMessenger_t* dbgMessenger, const nvimgcdcsDebugMessengerDesc_t* messengerDesc)
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