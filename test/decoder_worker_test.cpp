/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "../src/decoder_worker.h"
#include "mock_codec.h"
#include "mock_image_decoder.h"
#include "mock_image_decoder_factory.h"

namespace nvimgcdcs { namespace test {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Return;
using ::testing::TestWithParam;
using ::testing::Values;

using test_case_tuple_t = std::tuple<std::vector<nvimgcdcsBackendKind_t>, std::vector<nvimgcdcsBackend_t>, int, int>;

class DecoderWorkerTest : public TestWithParam<test_case_tuple_t>
{
  public:
    virtual ~DecoderWorkerTest() = default;

  protected:
    void SetUp() override
    {
        auto given_backend_kinds = std::get<0>(GetParam());
        allowed_backends_ = std::get<1>(GetParam());
        int start_index = std::get<2>(GetParam());
        expected_return_index_ = std::get<3>(GetParam());
        codec_ = std::make_unique<MockCodec>();
        EXPECT_CALL(*codec_.get(), getDecodersNum()).WillRepeatedly(Return(given_backend_kinds.size()));
        image_dec_factories_.resize(given_backend_kinds.size());
        image_decs_.resize(given_backend_kinds.size());
        image_dec_ptrs_.resize(given_backend_kinds.size());
        for (int i = 0; i < given_backend_kinds.size(); ++i) {
            auto backend_kind = given_backend_kinds[i];
            auto image_dec = std::make_unique<MockImageDecoder>();
            image_dec_ptrs_[i] = image_dec.get();
            image_dec_factories_[i] = new MockImageDecoderFactory();
            MockImageDecoderFactory* image_dec_factory(image_dec_factories_[i]);
            EXPECT_CALL(*image_dec_factory, getBackendKind()).WillRepeatedly(Return(backend_kind));
            EXPECT_CALL(*image_dec_factory, createDecoder(_, _, _)).WillRepeatedly(Return(ByMove(std::move(image_dec))));
            EXPECT_CALL(*codec_.get(), getDecoderFactory(i)).WillRepeatedly(Return(image_dec_factory));
        }

        decoder_worker_ = std::make_unique<DecoderWorker>(nullptr, 0, allowed_backends_, "", codec_.get(), start_index);
    }

    void TearDown() override
    {
        decoder_worker_.reset();
        image_dec_ptrs_.clear();
        for (auto f : image_dec_factories_) {
            delete f;
        }
        image_dec_factories_.clear();
        codec_.reset();
        allowed_backends_.clear();
    }

    std::unique_ptr<MockCodec> codec_;
    std::vector<std::unique_ptr<MockImageDecoder>> image_decs_;
    std::vector<IImageDecoder*> image_dec_ptrs_;
    std::vector<MockImageDecoderFactory*> image_dec_factories_;
    std::unique_ptr<DecoderWorker> decoder_worker_;
    std::vector<nvimgcdcsBackend_t> allowed_backends_;
    int expected_return_index_;
};

TEST_P(DecoderWorkerTest, for_given_backend_kinds_and_allowed_backends_get_decoder_returns_correct_decoder)
{
    IImageDecoder* decoder = decoder_worker_->getDecoder();
    if (expected_return_index_ == -1) {
        EXPECT_EQ(decoder, nullptr);
    } else {
        EXPECT_EQ(decoder, image_dec_ptrs_[expected_return_index_]);
    }
}

// clang-format off

namespace {
test_case_tuple_t test_cases[] = {
    //For given one CPU backend and allowed CPU backend, return index 0
    {{NVIMGCDCS_BACKEND_KIND_CPU_ONLY}, 
     {{NVIMGCDCS_STRUCTURE_TYPE_BACKEND, 0, NVIMGCDCS_BACKEND_KIND_CPU_ONLY, {NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS, 0, 50}}},
     0, 0
    },
    //For given one CPU backend and allowed all, return index 0
    {{NVIMGCDCS_BACKEND_KIND_CPU_ONLY}, 
     {},//all backends allowed
     0, 0
    },
    //For given 3 backends in order HW, GPU, CPU and allowed all, return index 0
    {{NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_CPU_ONLY}, 
    {},//all backends allowed
     0, 0
    },
    ///For given 3 backends in order HW, GPU, CPU, and allowed only HW, return index 0
    {{NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_CPU_ONLY}, 
    {{NVIMGCDCS_STRUCTURE_TYPE_BACKEND, 0, NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY, {NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS, 0, 50}}},
     0, 0
    },
    //For given 3 backends in order HW, GPU, CPU, and allowed only GPU, return index 1
    {{NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_CPU_ONLY}, 
    {{NVIMGCDCS_STRUCTURE_TYPE_BACKEND, 0, NVIMGCDCS_BACKEND_KIND_GPU_ONLY, {NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS, 0, 50}}},
     0, 1
    },
    //For given 3 backends in order HW,GPU,CPU and allowed CPU only, return decoder with index 2
    {{NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_CPU_ONLY}, 
    {{NVIMGCDCS_STRUCTURE_TYPE_BACKEND, 0, NVIMGCDCS_BACKEND_KIND_CPU_ONLY, {NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS, 0, 50}}},
     0, 2
    },
    //For given 3 backends in order HW,GPU,CPU, and allowed only HYBRID (which is not present), return nullptr
    {{NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_CPU_ONLY}, 
    {{NVIMGCDCS_STRUCTURE_TYPE_BACKEND, 0, NVIMGCDCS_BACKEND_KIND_HYBRID_CPU_GPU, {NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS, 0, 50}}},
     0, -1 //nullptr 
    },
    //For given 3 backends in order HW,GPU,CPU, and all allowed backends and start index 1 (first fallback), return decoder with index 1
    {{NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_CPU_ONLY}, 
    {},
     1, 1
    },
    //For given 3 backends in order HW,GPU,CPU, and allowed GPU and CPU, return decoder with index 1 (GPU)
    {{NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_CPU_ONLY}, 
    {{NVIMGCDCS_STRUCTURE_TYPE_BACKEND, 0, NVIMGCDCS_BACKEND_KIND_CPU_ONLY, {NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS, 0, 50}},
     {NVIMGCDCS_STRUCTURE_TYPE_BACKEND, 0, NVIMGCDCS_BACKEND_KIND_GPU_ONLY, {NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS, 0, 50}}
    },
     0, 1
    },
    //For given 3 backends in order HW,CPU, GPU, and allowed GPU and CPU, return decoder with index 1 (CPU)
    {{NVIMGCDCS_BACKEND_KIND_HW_GPU_ONLY, NVIMGCDCS_BACKEND_KIND_CPU_ONLY, NVIMGCDCS_BACKEND_KIND_GPU_ONLY}, 
    {{NVIMGCDCS_STRUCTURE_TYPE_BACKEND, 0, NVIMGCDCS_BACKEND_KIND_CPU_ONLY, {NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS, 0, 50}},
     {NVIMGCDCS_STRUCTURE_TYPE_BACKEND, 0, NVIMGCDCS_BACKEND_KIND_GPU_ONLY, {NVIMGCDCS_STRUCTURE_TYPE_BACKEND_PARAMS, 0, 50}}
    },
     0, 1
    },

};
// clang-format on
} // namespace

INSTANTIATE_TEST_SUITE_P(DECODER_WORKER_GET_DECODER_TEST, DecoderWorkerTest, ::testing::ValuesIn(test_cases));

}} // namespace nvimgcdcs::test