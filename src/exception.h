/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

namespace nvimgcodec {

enum Status
{
    STATUS_OK = 0,
    NOT_VALID_FORMAT_STATUS = 1,
    UNSUPPORTED_FORMAT_STATUS = 2,
    BAD_FORMAT_STATUS = 3,
    PARSE_STATUS = 4,
    ALLOCATION_ERROR = 5,
    INTERNAL_ERROR = 6,
    INVALID_PARAMETER = 7,
    CUDA_CALL_ERROR = 8,
    BAD_STATE = 9
};

const char* getErrorString(Status);

class Exception : public std::exception
{
  public:
    explicit Exception(
        Status eStatus, const std::string& rMessage = "", const std::string& rLoc = "");

    inline virtual ~Exception() throw() { ; }

    virtual const char* what() const throw();

    Status status() const;

    const char* message() const;

    const char* where() const;

  private:
    Exception();
    Status eStatus_;
    std::string sMessage_;
    std::string sLocation_;
};

#define FatalError(status, message)                     \
    {                                                   \
        std::stringstream _where;                       \
        _where << "At " << __FILE__ << ":" << __LINE__; \
        throw Exception(status, message, _where.str()); \
    }

#define CHECK_NULL(ptr)                                    \
    {                                                      \
        if (!ptr)                                          \
            FatalError(INVALID_PARAMETER, "null pointer"); \
    }


#define CHECK_CUDA(call)                                       \
    {                                                          \
        cudaError_t _e = (call);                               \
        if (_e != cudaSuccess) {                               \
            std::stringstream _error;                          \
            _error << "CUDA Runtime failure: '#" << std::to_string(_e) << "'"; \
            FatalError(CUDA_CALL_ERROR, _error.str());         \
        }                                                      \
    }

#define LOG_CUDA_ERROR(call)                                            \
    {                                                                   \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            std::stringstream _error;                                   \
            std::cerr << "CUDA Runtime failure: '#" << std::to_string(_e) << std::endl; \
        }                                                               \
    }

 #define CHECK_CU(call)                                                                             \
     {                                                                                              \
         CUresult _e = (call);                                                                      \
         if (_e != CUDA_SUCCESS) {                                                                  \
             std::stringstream _error;                                                              \
             _error << "CUDA Driver API failure: '#" << std::to_string(_e) << "'";                  \
             FatalError(CUDA_CALL_ERROR, _error.str());                                             \
         }                                                                                          \
     }

} // namespace nvimgcodec
