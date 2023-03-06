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

#if defined(__CUDACC__)
#define NVIMGCDCS_HOST_DEV __host__ __device__
#else
#define NVIMGCDCS_HOST_DEV
#endif

#if defined(__CUDACC__)
#define NVIMGCDCS_HOST __host__
#else
#define NVIMGCDCS_HOST
#endif

#if defined(__CUDACC__)
#define NVIMGCDCS_DEVICE __device__
#else
#define NVIMGCDCS_DEVICE
#endif
