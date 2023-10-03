
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

#include "nvimgcodec.h"

inline size_t sample_type_to_bytes_per_element(nvimgcodecSampleDataType_t sample_type)
{
    return static_cast<unsigned int>(sample_type)>> (8+3);
}