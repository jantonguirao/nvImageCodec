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

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../src/idecode_state.h"
#include "../src/icode_stream.h"
#include "../src/iimage.h"

namespace nvimgcodec { namespace test {

class MockImageDecoder : public IImageDecoder
{
  public:
    MOCK_METHOD(nvimgcodecBackendKind_t, getBackendKind, (), (const, override));
    MOCK_METHOD(std::unique_ptr<IDecodeState>, createDecodeStateBatch, (), (const, override));
    MOCK_METHOD(void, canDecode,
        (const std::vector<ICodeStream*>&, const std::vector<IImage*>&, const nvimgcodecDecodeParams_t*, std::vector<bool>*,
            std::vector<nvimgcodecProcessingStatus_t>*),
        (const, override));
    MOCK_METHOD(std::unique_ptr<ProcessingResultsFuture>, decode,
        (IDecodeState*, const std::vector<ICodeStream*>&, const std::vector<IImage*>&, const nvimgcodecDecodeParams_t*), (override));
    MOCK_METHOD(const char*, decoderId, (), (const, override));
};


}} // namespace nvimgcodec::test