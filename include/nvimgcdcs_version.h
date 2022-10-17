/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
// clang-format off
#ifndef NVIMGCDCS_VERSION_H__
#define NVIMGCDCS_VERSION_H__

#define NVIMGCDCS_VER_MAJOR 0
#define NVIMGCDCS_VER_MINOR 1
#define NVIMGCDCS_VER_PATCH 0
#define NVIMGCDCS_VER_BUILD 0


#define MAKE_SEMANTIC_VERSION(major, minor, patch) \
    ((major << 24) | (minor << 16) | patch)

#define NVIMGCDCS_VER                                               \
    MAKE_SEMANTIC_VERSION(NVIMGCDCS_VER_MAJOR, NVIMGCDCS_VER_MINOR, \
        NVIMGCDCS_VER_PATCH)

#endif // NVIMGCDCS_VERSION 
