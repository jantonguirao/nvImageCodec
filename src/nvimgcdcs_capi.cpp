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
#include "codec_registry.h"
#include "exception.h"
#include "plugin_framework.h"

#include <iostream>
#include <stdexcept>

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

nvimgcdcsStatus_t nvimgcdcsInstanceCreate(
    nvimgcdcsInstance_t* instance, nvimgcdcsInstanceCreateInfo_t createInfo)
{
    nvimgcdcsStatus_t ret         = NVIMGCDCS_STATUS_SUCCESS;
    nvimgcdcsInstance_t nvimgcdcs = nullptr;
    NVIMGCDCSAPI_TRY
    {
        CHECK_NULL(instance);
        nvimgcdcs = new nvimgcdcsHandle(createInfo.device_allocator, createInfo.pinned_allocator);
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
    nvimgcdcsCodeStream(CodecRegistry* codec_registry)
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
    nvimgcdcsCodeStream_t* stream_handle, const unsigned char* data, size_t size)
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
