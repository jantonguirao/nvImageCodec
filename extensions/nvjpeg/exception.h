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
#include <map>
#include <nvimgcodecs.h>
#include <nvjpeg.h>

const char* getErrorString(nvjpegStatus_t);

class NvJpegException : public std::exception
{
  public:
    explicit NvJpegException(
        nvjpegStatus_t eStatus, const std::string& rMessage = "", const std::string& rLoc = "");
    explicit NvJpegException(
        cudaError_t eStatus, const std::string& rMessage = "", const std::string& rLoc = "");

    inline virtual ~NvJpegException() throw() { ; }

    virtual const char* what() const throw();

    nvjpegStatus_t status() const;
    
    cudaError_t cudaStatus() const;

    const char* message() const;

    const char* where() const;

    std::string info() const throw();

    nvimgcdcsStatus_t nvimgcdcsStatus() const;

  private:
    NvJpegException();
    nvjpegStatus_t eStatus_;
    cudaError_t eCudaStatus_;
    bool isCudaStatus_;
    std::string sMessage_;
    std::string sLocation_;
};

#define FatalError(status, message)                     \
    {                                                   \
        std::stringstream _where;                       \
        _where << "At " << __FILE__ << ":" << __LINE__; \
        throw NvJpegException(status, message, _where.str()); \
    }     
