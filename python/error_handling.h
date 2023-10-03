/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#define CHECK_CUDA(call)                                                                                       \
    {                                                                                                          \
        cudaError_t _e = (call);                                                                               \
        if (_e != cudaSuccess) {                                                                               \
            std::stringstream _error;                                                                          \
            _error << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(_error.str());                                                            \
        }                                                                                                      \
    }

#define CHECK_NVIMGCODEC(call)                                   \
    {                                                           \
        nvimgcodecStatus_t _e = (call);                          \
        if (_e != NVIMGCODEC_STATUS_SUCCESS) {                   \
            std::stringstream _error;                           \
            _error << "nvImageCodec failure: '#" << _e << "'"; \
            throw std::runtime_error(_error.str());             \
        }                                                       \
    }
    
void check_cuda_buffer(const void* ptr);
