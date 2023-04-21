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

#define XM_CHECK_NULL(ptr)                      \
    {                                           \
        if (!ptr)                               \
            std::runtime_error("null pointer"); \
    }

#define XM_CHECK_CUDA(call)                                    \
    {                                                          \
        cudaError_t _e = (call);                               \
        if (_e != cudaSuccess) {                               \
            std::stringstream _error;                          \
            _error << "CUDA Runtime failure: '#" << _e << "'"; \
            throw std::runtime_error(_error.str());            \
        }                                                      \
    }
#define XM_CHECK_NVJPEG(call)                                    \
    {                                                            \
        nvjpegStatus_t _e = (call);                              \
        if (_e != NVJPEG_STATUS_SUCCESS) {                       \
            std::stringstream _error;                            \
            _error << "nvJpeg Runtime failure: '#" << _e << "'"; \
            throw std::runtime_error(_error.str());              \
        }                                                        \
    }