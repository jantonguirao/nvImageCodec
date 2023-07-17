/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <map>
#include "exception.h"
#pragma once

#define XM_CHECK_NULL(ptr)                                                 \
    {                                                                      \
        if (!ptr)                                                          \
            FatalError(NVJPEG2K_STATUS_INVALID_PARAMETER, "null pointer"); \
    }

#define XM_CHECK_CUDA(call)                                    \
    {                                                          \
        cudaError_t _e = (call);                               \
        if (_e != cudaSuccess) {                               \
            std::stringstream _error;                          \
            _error << "CUDA Runtime failure: '#" << _e << "'"; \
            FatalError(_e, _error.str());                      \
        }                                                      \
    }

#define XM_CHECK_NVJPEG2K(call)                                    \
    {                                                              \
        nvjpeg2kStatus_t _e = (call);                              \
        if (_e != NVJPEG2K_STATUS_SUCCESS) {                       \
            std::stringstream _error;                              \
            _error << "nvjpeg2k Runtime failure: '#" << _e << "'"; \
            FatalError(_e, _error.str());                          \
        }                                                          \
    }

#define XM_NVJPEG2K_D_LOG_DESTROY(call)                            \
    {                                                              \
        nvjpeg2kStatus_t _e = (call);                              \
        if (_e != NVJPEG2K_STATUS_SUCCESS) {                       \
            std::stringstream _error;                              \
            _error << "nvjpeg2k Runtime failure: '#" << _e << "'"; \
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, _error.str());    \
        }                                                          \
    }

#define XM_NVJPEG2K_E_LOG_DESTROY(call)                            \
    {                                                              \
        nvjpeg2kStatus_t _e = (call);                              \
        if (_e != NVJPEG2K_STATUS_SUCCESS) {                       \
            std::stringstream _error;                              \
            _error << "nvjpeg2k Runtime failure: '#" << _e << "'"; \
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, _error.str());    \
        }                                                          \
    }

#define XM_CUDA_LOG_DESTROY(call)                               \
    {                                                           \
        cudaError_t _e = (call);                                \
        if (_e != cudaSuccess) {                                \
            std::stringstream _error;                           \
            _error << "CUDA Runtime failure: '#" << _e << "'";  \
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, _error.str()); \
        }                                                       \
    }
