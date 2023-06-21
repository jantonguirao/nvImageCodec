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

#include <nvjpeg.h>

namespace nvjpeg {

int nvjpeg_flat_version(int major, int minor, int patch);
int nvjpeg_get_version();
bool nvjpeg_at_least(int major, int minor, int patch);
unsigned int get_nvjpeg_flags(const char* module_name, const char* options = "");

}