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

#include <nvimgcodec.h>
#include <nvjpeg2k.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

const char* getErrorString(nvjpeg2kStatus_t);

class NvJpeg2kException : public std::exception
{
  public:
    explicit NvJpeg2kException(nvjpeg2kStatus_t eStatus, const std::string& rMessage = "", const std::string& rLoc = "");
    explicit NvJpeg2kException(cudaError_t eStatus, const std::string& rMessage = "", const std::string& rLoc = "");

    inline virtual ~NvJpeg2kException() throw() { ; }

    virtual const char* what() const throw();

    nvjpeg2kStatus_t status() const;

    cudaError_t cudaStatus() const;

    const char* message() const;

    const char* where() const;

    std::string info() const throw();

    nvimgcodecStatus_t nvimgcodecStatus() const;

  private:
    NvJpeg2kException();
    nvjpeg2kStatus_t eStatus_;
    cudaError_t eCudaStatus_;
    bool isCudaStatus_;
    std::string sMessage_;
    std::string sLocation_;
};

#define FatalError(status, message)                             \
    {                                                           \
        std::stringstream _where;                               \
        _where << "At " << __FILE__ << ":" << __LINE__;         \
        throw NvJpeg2kException(status, message, _where.str()); \
    }
