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

#define XM_CHECK_NULL(ptr)                            \
    {                                                 \
        if (!ptr)                                     \
            throw std::runtime_error("null pointer"); \
    }

#define LIBTIFF_CALL_SUCCESS 1
#define LIBTIFF_CALL(call)                                                                                     \
    do {                                                                                                       \
        int retcode = (call);                                                                                  \
        if (LIBTIFF_CALL_SUCCESS != retcode)                                                                   \
            throw std::runtime_error("libtiff call failed with code " + std::to_string(retcode) + ": " #call); \
    } while (0)
