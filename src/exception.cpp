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

namespace nvimgcdcs {

namespace StringsNVIMGCDCS {
const std::string sNotValidFormat = "Not valid format";
const std::string sUnsupportedFormat = "Unsupported format";
const std::string sBadStream = "Corrupted stream";
const std::string sStreamParse = "Stream parse error";
const std::string sAllocationError = "Memory allocator error";
const std::string sInternalError = "Internal error";
const std::string sParameterError = "Error in the API call";
const std::string sCUDAError = "Error in the CUDA API call";
} // namespace StringsNVIMGCDCS

const char *getErrorString(StatusNVIMGCDCS eStatus_)
{
    switch (eStatus_) {
    case NOT_VALID_FORMAT_STATUS:
        return StringsNVIMGCDCS::sNotValidFormat.c_str();
    case UNSUPPORTED_FORMAT_STATUS:
        return StringsNVIMGCDCS::sUnsupportedFormat.c_str();
    case BAD_FORMAT_STATUS:
        return StringsNVIMGCDCS::sBadStream.c_str();
    case PARSE_STATUS:
        return StringsNVIMGCDCS::sStreamParse.c_str();
    case ALLOCATION_ERROR:
        return StringsNVIMGCDCS::sAllocationError.c_str();
    case INVALID_PARAMETER:
        return StringsNVIMGCDCS::sParameterError.c_str();
    case CUDA_CALL_ERROR:
        return StringsNVIMGCDCS::sCUDAError.c_str();
    case INTERNAL_ERROR:
    default:
        return StringsNVIMGCDCS::sInternalError.c_str();
    }
}

ExceptionNVIMGCDCS::ExceptionNVIMGCDCS(StatusNVIMGCDCS eStatus, const std::string &rMessage,
                                       const std::string &rLoc)
    : eStatus_(eStatus), sMessage_(rMessage), sLocation_(rLoc)
{
    ;
}

const char *ExceptionNVIMGCDCS::what() const throw() { return getErrorString(eStatus_); };

StatusNVIMGCDCS ExceptionNVIMGCDCS::status() const { return eStatus_; }

const char *ExceptionNVIMGCDCS::message() const { return sMessage_.c_str(); }

const char *ExceptionNVIMGCDCS::where() const { return sLocation_.c_str(); }

} // namespace nvimgcdcs
