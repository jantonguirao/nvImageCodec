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

#include <nvimgcodec.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "../src/iimage_encoder.h"
#include "../src/iimage_encoder_factory.h"

namespace nvimgcodec {
namespace test {

class MockImageEncoderFactory : public IImageEncoderFactory
{
  public:
    MOCK_METHOD(std::string, getEncoderId, (), (const, override));
    MOCK_METHOD(std::string, getCodecName, (), (const, override));
    MOCK_METHOD(nvimgcodecBackendKind_t, getBackendKind, (), (const, override));
    MOCK_METHOD(std::unique_ptr<IImageEncoder>, createEncoder,
        (const nvimgcodecExecutionParams_t*, const char*), (const, override));
};

}}