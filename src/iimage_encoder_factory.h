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

#include <nvimgcodecs.h>
#include <memory>
#include <string>

namespace nvimgcdcs {

class IImageEncoder;

class IImageEncoderFactory
{
  public:
    virtual ~IImageEncoderFactory() = default;
    virtual std::string getEncoderId() const = 0;
    virtual std::string getCodecName() const = 0;
    virtual nvimgcdcsBackendKind_t getBackendKind() const = 0;
    virtual std::unique_ptr<IImageEncoder> createEncoder(
        int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options) const = 0;
};
} // namespace nvimgcdcs