/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "exception.h"

const std::map<nvjpeg2kStatus_t, nvimgcdcsStatus_t> nvjpeg2k_to_nvimgcodecs_error_map = 
    {{NVJPEG2K_STATUS_SUCCESS, NVIMGCDCS_STATUS_SUCCESS},    
    {NVJPEG2K_STATUS_NOT_INITIALIZED, NVIMGCDCS_EXTENSION_STATUS_NOT_INITIALIZED},
    {NVJPEG2K_STATUS_INVALID_PARAMETER, NVIMGCDCS_EXTENSION_STATUS_INVALID_PARAMETER},
    {NVJPEG2K_STATUS_INTERNAL_ERROR, NVIMGCDCS_EXTENSION_STATUS_INTERNAL_ERROR},    
    {NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED, NVIMGCDCS_EXTENSION_STATUS_IMPLEMENTATION_NOT_SUPPORTED},
    {NVJPEG2K_STATUS_EXECUTION_FAILED, NVIMGCDCS_EXTENSION_STATUS_EXECUTION_FAILED},
    {NVJPEG2K_STATUS_BAD_JPEG, NVIMGCDCS_EXTENSION_STATUS_BAD_CODE_STREAM},    
    {NVJPEG2K_STATUS_ALLOCATOR_FAILURE, NVIMGCDCS_EXTENSION_STATUS_ALLOCATOR_FAILURE},
    {NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED, NVIMGCDCS_EXTENSION_STATUS_CODESTREAM_UNSUPPORTED}};

namespace StatusStrings {
const std::string sExtNotInit        = "nvjpeg2k extension: not initialized";
const std::string sExtInvalidParam   = "nvjpeg2k extension: Invalid parameter";
const std::string sExtBadJpeg        = "nvjpeg2k extension: Bad jpeg";
const std::string sExtJpegUnSupp     = "nvjpeg2k extension: Jpeg not supported";
const std::string sExtAllocFail      = "nvjpeg2k extension: allocator failure";
const std::string sExtIntErr         = "nvjpeg2k extension: internal error";
const std::string sExtImplNA         = "nvjpeg2k extension: implementation not supported";
const std::string sExtExeFailed      = "nvjpeg2k extension: execution failed";
const std::string sExtCudaCallError  = "nvjpeg2k extension: cuda call error";
} // namespace StatusStrings

const char* getErrorString(nvjpeg2kStatus_t eStatus_)
{
    switch (eStatus_) {        
    case NVJPEG2K_STATUS_ALLOCATOR_FAILURE:
        return StatusStrings::sExtAllocFail.c_str();
    case NVJPEG2K_STATUS_BAD_JPEG:
        return StatusStrings::sExtBadJpeg.c_str();      
    case NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED:
        return StatusStrings::sExtJpegUnSupp.c_str();
    case NVJPEG2K_STATUS_EXECUTION_FAILED:
        return StatusStrings::sExtExeFailed.c_str();
    case NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return StatusStrings::sExtImplNA.c_str();
    case NVJPEG2K_STATUS_INTERNAL_ERROR:
        return StatusStrings::sExtIntErr.c_str();
    case NVJPEG2K_STATUS_INVALID_PARAMETER:
        return StatusStrings::sExtInvalidParam.c_str();
    case NVJPEG2K_STATUS_NOT_INITIALIZED:
        return StatusStrings::sExtNotInit.c_str();
    default:
        return StatusStrings::sExtIntErr.c_str();

    }
}

NvJpeg2kException::NvJpeg2kException(nvjpeg2kStatus_t eStatus, const std::string& rMessage, const std::string& rLoc)
    : eStatus_(eStatus)
    , sMessage_(rMessage)
    , sLocation_(rLoc)
{
    ;
}

NvJpeg2kException::NvJpeg2kException(cudaError_t eCudaStatus, const std::string& rMessage, const std::string& rLoc)
    : eCudaStatus_(eCudaStatus)
    , sMessage_(rMessage)
    , sLocation_(rLoc)
{
    isCudaStatus_ = true;;
}

const char* NvJpeg2kException::what() const throw()
{
    if (isCudaStatus_)
        return StatusStrings::sExtCudaCallError.c_str();
    else
        return getErrorString(eStatus_);
};

nvjpeg2kStatus_t NvJpeg2kException::status() const
{    
    return eStatus_;
}

cudaError_t NvJpeg2kException::cudaStatus() const
{
    return eCudaStatus_;
}

const char* NvJpeg2kException::message() const
{
    return sMessage_.c_str();
}

const char* NvJpeg2kException::where() const
{
    return sLocation_.c_str();
}

std::string NvJpeg2kException::info() const throw()
{   
    std::string info(getErrorString(eStatus_)); 
    if (isCudaStatus_)
        info = StatusStrings::sExtCudaCallError;        
    return info + " " + sLocation_;   
}

nvimgcdcsStatus_t NvJpeg2kException::nvimgcdcsStatus() const
{    
    if (isCudaStatus_)
        return NVIMGCDCS_EXTENSION_STATUS_CUDA_CALL_ERROR;
    else
    {
        auto it = nvjpeg2k_to_nvimgcodecs_error_map.find(eStatus_);
        if (it != nvjpeg2k_to_nvimgcodecs_error_map.end())
            return it->second;
        else
            return NVIMGCDCS_EXTENSION_STATUS_INTERNAL_ERROR;
    }
}
