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
#include "image_generic_decoder.h"
#include "image_generic_encoder.h"
#include "iostream_factory.h"
#include "log.h"
#include "nvimgcodecs_director.h"
#include "plugin_framework.h"
#include "processing_results.h"
#include "nvimgcodecs_type_utils.h"
#include "file_ext_codec.h"

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
    std::unique_ptr<ImageGenericDecoder> image_decoder_;
};

struct nvimgcdcsEncoder
{
    nvimgcdcsInstance_t instance_;
    std::unique_ptr<ImageGenericEncoder> image_encoder_;
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
    nvimgcdcsInstance_t nvimgcdcs_instance_;
    Image image_;
};

nvimgcdcsStatus_t nvimgcdcsGetProperties(nvimgcdcsProperties_t* properties)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(properties);
            if (properties->type != NVIMGCDCS_STRUCTURE_TYPE_PROPERTIES) {
                return NVIMGCDCS_STATUS_INVALID_PARAMETER;
            }
            properties->version = NVIMGCDCS_VER;
            properties->ext_api_version = NVIMGCDCS_EXT_API_VER;
            properties->cudart_version = CUDART_VERSION;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

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

            ret = extension->nvimgcdcs_instance_->director_.plugin_framework_.unregisterExtension(extension->extension_ext_handle_);
            delete extension;
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

static nvimgcdcsStatus_t nvimgcdcsStreamCreate(nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* code_stream)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    nvimgcdcsCodeStream_t stream = nullptr;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(code_stream);
            stream = new nvimgcdcsCodeStream(&instance->director_.codec_registry_);
            *code_stream = stream;
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
    nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* code_stream, const char* file_name)
{
    nvimgcdcsStatus_t ret = nvimgcdcsStreamCreate(instance, code_stream);

    NVIMGCDCSAPI_TRY
        {
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                (*code_stream)->code_stream_.parseFromFile(std::string(file_name));
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateFromHostMem(
    nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* code_stream, const unsigned char* data, size_t size)
{
    nvimgcdcsStatus_t ret = nvimgcdcsStreamCreate(instance, code_stream);

    NVIMGCDCSAPI_TRY
        {
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                (*code_stream)->code_stream_.parseFromMem(data, size);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToFile(nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* code_stream, const char* file_name, const nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = nvimgcdcsStreamCreate(instance, code_stream);
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(code_stream)
            CHECK_NULL(file_name)
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                (*code_stream)->code_stream_.setOutputToFile(file_name);
                (*code_stream)->code_stream_.setImageInfo(image_info);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamCreateToHostMem(nvimgcdcsInstance_t instance, nvimgcdcsCodeStream_t* code_stream,
    void* ctx, nvimgcdcsGetBufferFunc_t get_buffer_func, const nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = nvimgcdcsStreamCreate(instance, code_stream);
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(code_stream)
            CHECK_NULL(image_info)
            CHECK_NULL(get_buffer_func)
            if (ret == NVIMGCDCS_STATUS_SUCCESS) {
                (*code_stream)->code_stream_.setOutputToHostMem(ctx, get_buffer_func);
                (*code_stream)->code_stream_.setImageInfo(image_info);
            }
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamDestroy(nvimgcdcsCodeStream_t code_stream)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(code_stream)
            delete code_stream;
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamGetImageInfo(nvimgcdcsCodeStream_t code_stream, nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(code_stream)
            CHECK_NULL(image_info)
            return code_stream->code_stream_.getImageInfo(image_info);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsCodeStreamSetImageInfo(nvimgcdcsCodeStream_t code_stream, nvimgcdcsImageInfo_t* image_info)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(code_stream)
            CHECK_NULL(image_info)
            code_stream->code_stream_.setImageInfo(image_info);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderCreate(
    nvimgcdcsInstance_t instance, nvimgcdcsDecoder_t* decoder, int device_id, const char* options)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(decoder)
            if (device_id == -1)
                CHECK_CUDA(cudaGetDevice(&device_id));
            std::unique_ptr<ImageGenericDecoder> image_decoder = instance->director_.createGenericDecoder(device_id, options);
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

NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsDecoderCanDecode(nvimgcdcsDecoder_t decoder, const nvimgcdcsCodeStream_t* streams,
    const nvimgcdcsImage_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params, nvimgcdcsProcessingStatus_t* processing_status,
    bool force_format)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(decoder)
            CHECK_NULL(streams)
            CHECK_NULL(images)
            CHECK_NULL(params)
            std::vector<nvimgcdcs::ICodeStream*> internal_code_streams;
            std::vector<nvimgcdcs::IImage*> internal_images;

            for (int i = 0; i < batch_size; ++i) {
                internal_code_streams.push_back(&streams[i]->code_stream_);
                internal_images.push_back(&images[i]->image_);
            }

            decoder->image_decoder_->canDecode(internal_code_streams, internal_images, params, processing_status, force_format);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsDecoderDecode(nvimgcdcsDecoder_t decoder, const nvimgcdcsCodeStream_t* streams, const nvimgcdcsImage_t* images,
    int batch_size, const nvimgcdcsDecodeParams_t* params, nvimgcdcsFuture_t* future)
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

            (*future)->handle_ = std::move(decoder->image_decoder_->decode(internal_code_streams, internal_images, params));
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

NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderCreate(
    nvimgcdcsInstance_t instance, nvimgcdcsEncoder_t* encoder, int device_id, const char* options)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(instance)
            CHECK_NULL(encoder)
            if (device_id == -1)
                CHECK_CUDA(cudaGetDevice(&device_id));
            std::unique_ptr<ImageGenericEncoder> image_encoder = instance->director_.createGenericEncoder(device_id, options);
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

NVIMGCDCSAPI nvimgcdcsStatus_t nvimgcdcsEncoderCanEncode(nvimgcdcsEncoder_t encoder, const nvimgcdcsImage_t* images,
    const nvimgcdcsCodeStream_t* streams, int batch_size, const nvimgcdcsEncodeParams_t* params, nvimgcdcsProcessingStatus_t* processing_status,
    bool force_format)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;

    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(encoder)
            CHECK_NULL(streams)
            CHECK_NULL(images)
            CHECK_NULL(params)

            std::vector<nvimgcdcs::ICodeStream*> internal_code_streams;
            std::vector<nvimgcdcs::IImage*> internal_images;

            for (int i = 0; i < batch_size; ++i) {
                internal_code_streams.push_back(&streams[i]->code_stream_);
                internal_images.push_back(&images[i]->image_);
            }

            encoder->image_encoder_->canEncode(internal_images, internal_code_streams, params, processing_status, force_format);
        }
    NVIMGCDCSAPI_CATCH(ret)
    return ret;
}

nvimgcdcsStatus_t nvimgcdcsEncoderEncode(nvimgcdcsEncoder_t encoder, const nvimgcdcsImage_t* images, const nvimgcdcsCodeStream_t* streams,
    int batch_size, const nvimgcdcsEncodeParams_t* params, nvimgcdcsFuture_t* future)
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

            (*future)->handle_ = std::move(encoder->image_encoder_->encode(internal_images, internal_code_streams, params));
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

nvimgcdcsStatus_t nvimgcdcsFutureGetProcessingStatus(nvimgcdcsFuture_t future, nvimgcdcsProcessingStatus_t* processing_status, size_t* size)
{
    nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
    NVIMGCDCSAPI_TRY
        {
            CHECK_NULL(future)
            CHECK_NULL(size)
            std::vector<ProcessingResult> results(std::move(future->handle_->getAllCopy()));
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
