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

const std::map<nvjpegStatus_t, nvimgcdcsStatus_t> nvjpeg_status_to_nvimgcodecs_error_map = 
    {{NVJPEG_STATUS_SUCCESS, NVIMGCDCS_STATUS_SUCCESS},    
    {NVJPEG_STATUS_NOT_INITIALIZED, NVIMGCDCS_EXTENSION_STATUS_NOT_INITIALIZED},
    {NVJPEG_STATUS_INVALID_PARAMETER, NVIMGCDCS_EXTENSION_STATUS_INVALID_PARAMETER},
    {NVJPEG_STATUS_INTERNAL_ERROR, NVIMGCDCS_EXTENSION_STATUS_INTERNAL_ERROR},
    {NVJPEG_STATUS_INCOMPLETE_BITSTREAM, NVIMGCDCS_EXTENSION_STATUS_INCOMPLETE_BITSTREAM},
    {NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED, NVIMGCDCS_EXTENSION_STATUS_IMPLEMENTATION_NOT_SUPPORTED},
    {NVJPEG_STATUS_EXECUTION_FAILED, NVIMGCDCS_EXTENSION_STATUS_EXECUTION_FAILED},
    {NVJPEG_STATUS_BAD_JPEG, NVIMGCDCS_EXTENSION_STATUS_BAD_CODE_STREAM},
    {NVJPEG_STATUS_ARCH_MISMATCH, NVIMGCDCS_EXTENSION_STATUS_ARCH_MISMATCH},
    {NVJPEG_STATUS_ALLOCATOR_FAILURE, NVIMGCDCS_EXTENSION_STATUS_ALLOCATOR_FAILURE},
    {NVJPEG_STATUS_JPEG_NOT_SUPPORTED, NVIMGCDCS_EXTENSION_STATUS_CODESTREAM_UNSUPPORTED}};

namespace StatusStrings {
const std::string sExtNotInit        = "nvjpeg extension: not initialized";
const std::string sExtInvalidParam   = "nvjpeg extension: Invalid parameter";
const std::string sExtBadJpeg        = "nvjpeg extension: Bad jpeg";
const std::string sExtJpegUnSupp     = "nvjpeg extension: Jpeg not supported";
const std::string sExtAllocFail      = "nvjpeg extension: allocator failure";
const std::string sExtArchMis        = "nvjpeg extension: arch mismatch";
const std::string sExtIntErr         = "nvjpeg extension: internal error";
const std::string sExtImplNA         = "nvjpeg extension: implementation not supported";
const std::string sExtIncBits        = "nvjpeg extension: incomplete bitstream";
const std::string sExtExeFailed      = "nvjpeg extension: execution failed";
const std::string sExtCudaCallError  = "nvjpeg extension: cuda call error";
} // namespace StatusStrings

const char* getErrorString(nvjpegStatus_t eStatus_)
{
    switch (eStatus_) {        
    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
        return StatusStrings::sExtAllocFail.c_str();
    case NVJPEG_STATUS_ARCH_MISMATCH:
        return StatusStrings::sExtArchMis.c_str();
    case NVJPEG_STATUS_BAD_JPEG:
        return StatusStrings::sExtBadJpeg.c_str();      
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
        return StatusStrings::sExtJpegUnSupp.c_str();
    case NVJPEG_STATUS_EXECUTION_FAILED:
        return StatusStrings::sExtExeFailed.c_str();
    case NVJPEG_STATUS_INCOMPLETE_BITSTREAM:
        return StatusStrings::sExtIncBits.c_str();
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return StatusStrings::sExtImplNA.c_str();
    case NVJPEG_STATUS_INTERNAL_ERROR:
        return StatusStrings::sExtIntErr.c_str();
    case NVJPEG_STATUS_INVALID_PARAMETER:
        return StatusStrings::sExtInvalidParam.c_str();
    case NVJPEG_STATUS_NOT_INITIALIZED:
        return StatusStrings::sExtNotInit.c_str();
    default:
        return StatusStrings::sExtIntErr.c_str();

    }
}

Exception::Exception(nvjpegStatus_t eStatus, const std::string& rMessage, const std::string& rLoc)
    : eStatus_(eStatus)
    , sMessage_(rMessage)
    , sLocation_(rLoc)
{
    ;
}

Exception::Exception(cudaError_t eCudaStatus, const std::string& rMessage, const std::string& rLoc)
    : eCudaStatus_(eCudaStatus)
    , sMessage_(rMessage)
    , sLocation_(rLoc)
{
    isCudaStatus_ = true;;
}

const char* Exception::what() const throw()
{
    if (isCudaStatus_)
        return StatusStrings::sExtCudaCallError.c_str();
    else
        return getErrorString(eStatus_);
};

nvjpegStatus_t Exception::status() const
{    
    return eStatus_;
}

cudaError_t Exception::cudaStatus() const
{
    return eCudaStatus_;
}

const char* Exception::message() const
{
    return sMessage_.c_str();
}

const char* Exception::where() const
{
    return sLocation_.c_str();
}

std::string Exception::info() const throw()
{   
    std::string info(getErrorString(eStatus_)); 
    if (isCudaStatus_)
        info = StatusStrings::sExtCudaCallError;        
    return info + " " + sLocation_;   
}

nvimgcdcsStatus_t Exception::nvimgcdcsStatus() const
{    
    if (isCudaStatus_)
        return NVIMGCDCS_EXTENSION_STATUS_CUDA_CALL_ERROR;
    else
    {
        auto it = nvjpeg_status_to_nvimgcodecs_error_map.find(eStatus_);
        if (it != nvjpeg_status_to_nvimgcodecs_error_map.end())
            return it->second;
        else
            return NVIMGCDCS_EXTENSION_STATUS_INTERNAL_ERROR;
    }
}
