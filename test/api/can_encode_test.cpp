/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nvimgcodecs.h>

#include "can_de_en_code_common.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

namespace nvimgcdcs { namespace test {

namespace {

using test_case_tuple_t =
    std::tuple<const std::vector<std::vector<nvimgcdcsProcessingStatus_t>>*, bool, const std::vector<nvimgcdcsProcessingStatus_t>*>;

class MockEncoderPlugin
{
  public:
    explicit MockEncoderPlugin(const nvimgcdcsFrameworkDesc_t framework, const std::vector<nvimgcdcsProcessingStatus_t>& return_status)
        : return_status_(return_status)
        , encoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_ENCODER_DESC, NULL,
              this,                // instance
              "mock_test_encoder", // id
              0x00000100,          // version
              "bmp",               // codec_type
              static_create, static_destroy, static_get_capabilities, static_can_encode, static_encode_batch}
    {
    }
    nvimgcdcsEncoderDesc_t getEncoderDesc() { return &encoder_desc_; }

  private:
    static nvimgcdcsStatus_t static_create(void* instance, nvimgcdcsEncoder_t* encoder, int device_id)
    {
        *encoder = static_cast<nvimgcdcsEncoder_t>(instance);
        return NVIMGCDCS_STATUS_SUCCESS;
    }
    static nvimgcdcsStatus_t static_destroy(nvimgcdcsEncoder_t encoder) { return NVIMGCDCS_STATUS_SUCCESS; }
    static nvimgcdcsStatus_t static_get_capabilities(nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
    {
        *size = 0;
        return NVIMGCDCS_STATUS_SUCCESS;
    }
    static nvimgcdcsStatus_t static_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
        nvimgcdcsImageDesc_t* images, nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
    {
        auto handle = reinterpret_cast<MockEncoderPlugin*>(encoder);
        nvimgcdcsProcessingStatus_t* s = status;
        for (int i = 0; i < batch_size; ++i) {
            *s = handle->return_status_[i];
            s++;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }
    static nvimgcdcsStatus_t static_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t* images,
        nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
    {
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    struct nvimgcdcsEncoderDesc encoder_desc_;
    const std::vector<nvimgcdcsProcessingStatus_t>& return_status_;
};

struct MockCodecExtensionFactory
{
  public:
    explicit MockCodecExtensionFactory(const std::vector<std::vector<nvimgcdcsProcessingStatus_t>>* statuses)
        : desc_{NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC, nullptr, this, "test_extension", 0x00000100, static_extension_create,
              static_extension_destroy}
        , statuses_(statuses)

    {
    }

    nvimgcdcsExtensionDesc_t* getExtensionDesc() { return &desc_; };

    struct Extension
    {
        explicit Extension(const nvimgcdcsFrameworkDesc_t framework, const std::vector<std::vector<nvimgcdcsProcessingStatus_t>>* statuses)
            : framework_(framework)
            , statuses_(statuses)
        {
            for (auto& item : *statuses_) {
                encoders_.emplace_back(framework, item);
                framework->registerEncoder(framework->instance, encoders_.back().getEncoderDesc());
            }
        }

        const nvimgcdcsFrameworkDesc_t framework_;
        std::vector<MockEncoderPlugin> encoders_;
        const std::vector<std::vector<nvimgcdcsProcessingStatus_t>>* statuses_;
    };

    static nvimgcdcsStatus_t static_extension_create(
        void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
    {
        auto handle = reinterpret_cast<MockCodecExtensionFactory*>(instance);
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new Extension(framework, handle->statuses_));
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    static nvimgcdcsStatus_t static_extension_destroy(nvimgcdcsExtension_t extension)
    {
        auto ext_handle = reinterpret_cast<Extension*>(extension);
        delete ext_handle;
        return NVIMGCDCS_STATUS_SUCCESS;
    }

  private:
    nvimgcdcsExtensionDesc_t desc_;
    const std::vector<std::vector<nvimgcdcsProcessingStatus_t>>* statuses_;
};

} // namespace

class NvImageCodecsCanEncodeApiTest : public TestWithParam<std::tuple<test_case_tuple_t, bool> >
{
  public:
    NvImageCodecsCanEncodeApiTest() {}
    virtual ~NvImageCodecsCanEncodeApiTest() = default;

  protected:
    void SetUp() override
    {
        test_case_tuple_t test_case = std::get<0>(GetParam());
        mock_extension_ = std::make_unique<MockCodecExtensionFactory>(std::get<0>(test_case));
        force_format_ = std::get<1>(test_case);
        expected_statuses_ = std::get<2>(test_case);
        register_extension_ = std::get<1>(GetParam());

        nvimgcdcsInstanceCreateInfo_t create_info;
        create_info.type = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.next = nullptr;
        create_info.device_allocator = nullptr;
        create_info.pinned_allocator = nullptr;
        create_info.load_extension_modules = false;
        create_info.executor = nullptr;
        create_info.num_cpu_threads = 1;

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceCreate(&instance_, create_info));
        if (register_extension_) {
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &extension_, mock_extension_->getExtensionDesc()));
        }
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderCreate(instance_, &encoder_, NVIMGCDCS_DEVICE_CURRENT));
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS, 0};
        image_info_ = {NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        image_info_.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;
        out_buffer_.resize(1);
        image_info_.buffer = out_buffer_.data();
        image_info_.buffer_size = 1;

        images_.clear();
        streams_.clear();

        for (size_t i = 0; i < expected_statuses_->size(); ++i) {
            nvimgcdcsCodeStream_t code_stream = nullptr;
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
                nvimgcdcsCodeStreamCreateToHostMem(instance_, &code_stream, out_buffer_.data(), out_buffer_.size(), "bmp", &image_info_));
            streams_.push_back(code_stream);
            nvimgcdcsImage_t image;
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &image, &image_info_));
            images_.push_back(image);
        }
    }

    virtual void TearDown()
    {
        for (auto im : images_) {
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageDestroy(im));
        }
        for (auto cs : streams_) {
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(cs));
        }
        if (encoder_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderDestroy(encoder_));
        if (extension_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(extension_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceDestroy(instance_));
        mock_extension_.reset();
    }

    nvimgcdcsInstance_t instance_;
    nvimgcdcsExtension_t extension_ = nullptr;
    std::unique_ptr<MockCodecExtensionFactory> mock_extension_;
    std::vector<unsigned char> out_buffer_;
    nvimgcdcsImageInfo_t image_info_;
    nvimgcdcsEncoder_t encoder_;
    nvimgcdcsEncodeParams_t params_;
    std::vector<nvimgcdcsImage_t> images_;
    std::vector<nvimgcdcsCodeStream_t> streams_;
    bool force_format_ = true;
    bool register_extension_ = true;
    const std::vector<nvimgcdcsProcessingStatus_t>* expected_statuses_;
};

TEST_P(NvImageCodecsCanEncodeApiTest, CanEncode)
{
    std::vector<nvimgcdcsProcessingStatus_t> processing_statuses(expected_statuses_->size());
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderCanEncode(encoder_, images_.data(), streams_.data(), streams_.size(), &params_,
                                            processing_statuses.data(), force_format_));
    for (size_t i = 0; i < streams_.size(); ++i) {
        if (register_extension_) {
            EXPECT_EQ((*expected_statuses_)[i], processing_statuses[i]);
        } else {
            EXPECT_EQ(NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED, processing_statuses[i]);
        }
    }
}

test_case_tuple_t can_encode_test_cases[] = {
    {&statuses_to_return_case1_with_force_format_true, true, &statuses_to_expect_for_case1_with_force_format_true},
    {&statuses_to_return_case1_with_force_format_false, false, &statuses_to_expect_for_case1_with_force_format_false},
    {&statuses_to_return_case2_with_force_format_true, true, &statuses_to_expect_for_case2_with_force_format_true},
    {&statuses_to_return_case2_with_force_format_false, false, &statuses_to_expect_for_case2_with_force_format_false},
    {&statuses_to_return_case3_with_force_format_true, true, &statuses_to_expect_for_case3_with_force_format_true},
    {&statuses_to_return_case3_with_force_format_false, false, &statuses_to_expect_for_case3_with_force_format_false},
    {&statuses_to_return_case4_with_force_format_true, true, &statuses_to_expect_for_case4_with_force_format_true},
    {&statuses_to_return_case4_with_force_format_false, false, &statuses_to_expect_for_case4_with_force_format_false}};
// clang-format on

INSTANTIATE_TEST_SUITE_P(
    API_CAN_ENCODE, NvImageCodecsCanEncodeApiTest, Combine(::testing::ValuesIn(can_encode_test_cases), ::testing::Values(true, false)));

}} // namespace nvimgcdcs::test
