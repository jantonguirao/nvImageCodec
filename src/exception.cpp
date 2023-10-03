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

namespace nvimgcodec {

namespace StatusStrings {
const std::string sNotValidFormat    = "Not valid format";
const std::string sUnsupportedFormat = "Unsupported format";
const std::string sBadStream         = "Corrupted stream";
const std::string sStreamParse       = "Stream parse error";
const std::string sAllocationError   = "Memory allocator error";
const std::string sInternalError     = "Internal error";
const std::string sParameterError    = "Error in the API call";
const std::string sCUDAError         = "Error in the CUDA API call";
} // namespace StatusStrings

const char* getErrorString(Status eStatus_)
{
    switch (eStatus_) {
    case NOT_VALID_FORMAT_STATUS:
        return StatusStrings::sNotValidFormat.c_str();
    case UNSUPPORTED_FORMAT_STATUS:
        return StatusStrings::sUnsupportedFormat.c_str();
    case BAD_FORMAT_STATUS:
        return StatusStrings::sBadStream.c_str();
    case PARSE_STATUS:
        return StatusStrings::sStreamParse.c_str();
    case ALLOCATION_ERROR:
        return StatusStrings::sAllocationError.c_str();
    case INVALID_PARAMETER:
        return StatusStrings::sParameterError.c_str();
    case CUDA_CALL_ERROR:
        return StatusStrings::sCUDAError.c_str();
    case INTERNAL_ERROR:
    default:
        return StatusStrings::sInternalError.c_str();
    }
}

Exception::Exception(Status eStatus, const std::string& rMessage, const std::string& rLoc)
    : eStatus_(eStatus)
    , sMessage_(rMessage)
    , sLocation_(rLoc)
{
    ;
}

const char* Exception::what() const throw()
{
    return getErrorString(eStatus_);
};

Status Exception::status() const
{
    return eStatus_;
}

const char* Exception::message() const
{
    return sMessage_.c_str();
}

const char* Exception::where() const
{
    return sLocation_.c_str();
}

} // namespace nvimgcodec
