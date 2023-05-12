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
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <nvimgcodecs.h>

#include <extensions/nvjpeg2k/nvjpeg2k_ext.h>
#include <parsers/parser_test_utils.h>

#include "nvjpeg2k_ext_test_common.h"

#include <test_utils.h>
#include "nvimgcodecs_tests.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

#define NV_DEVELOPER_DUMP_OUTPUT_IMAGE_TO_BMP 1

namespace nvimgcdcs { namespace test {

class NvJpeg2kExtDecoderTestBase : public NvJpeg2kExtTestBase
{
  public:
    virtual ~NvJpeg2kExtDecoderTestBase() = default;

    void SetUp()
    {
        NvJpeg2kExtTestBase::SetUp();
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderCreate(instance_, &decoder_, NVIMGCDCS_DEVICE_CURRENT));
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, 0};
        params_.enable_color_conversion = true;
    }

    void TearDown()
    {
        if (decoder_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderDestroy(decoder_));
        NvJpeg2kExtTestBase::TearDown();
    }

    nvimgcdcsDecoder_t decoder_;
    nvimgcdcsDecodeParams_t params_;
};

class NvJpeg2kExtDecoderTestSingleImage : public NvJpeg2kExtDecoderTestBase, public NvJpeg2kTestBase, public TestWithParam<const char*>
{
  public:
    virtual ~NvJpeg2kExtDecoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        image_file_ = GetParam();
        NvJpeg2kExtDecoderTestBase::SetUp();
        NvJpeg2kTestBase::SetUp();
    }

    virtual void TearDown()
    {
        NvJpeg2kTestBase::TearDown();
        NvJpeg2kExtDecoderTestBase::TearDown();
    }

    std::string image_file_;
};

TEST_P(NvJpeg2kExtDecoderTestSingleImage, SingleImage)
{
    LoadImageFromFilename(instance_, in_code_stream_, resources_dir + image_file_);

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(in_code_stream_, &image_info_));
    image_info_.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
    image_info_.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
    image_info_.num_planes = 3;
    for (int p = 0; p < image_info_.num_planes; p++) {
        image_info_.plane_info[p].height = image_info_.plane_info[0].height;
        image_info_.plane_info[p].width = image_info_.plane_info[0].width;
        image_info_.plane_info[p].row_stride = image_info_.plane_info[0].width;
        image_info_.plane_info[p].num_channels = 1;
        image_info_.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
    }
    image_info_.buffer_size = image_info_.plane_info[0].height * image_info_.plane_info[0].width * image_info_.num_planes;
    out_buffer_.resize(image_info_.buffer_size);
    image_info_.buffer = out_buffer_.data();
    image_info_.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &out_image_, &image_info_));
    streams_.push_back(in_code_stream_);
    images_.push_back(out_image_);
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderDecode(decoder_, streams_.data(), images_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    cudaDeviceSynchronize();
    nvimgcdcsProcessingStatus_t status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);
    decodeReference(resources_dir + '/' + image_file_);
    if (NV_DEVELOPER_DUMP_OUTPUT_IMAGE_TO_BMP) {
        write_bmp("./out.bmp", out_buffer_.data(), image_info_.plane_info[0].width,
            out_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width), image_info_.plane_info[0].width,
            out_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width) * 2, image_info_.plane_info[0].width,
            image_info_.plane_info[0].width, image_info_.plane_info[0].height);

        write_bmp("./ref.bmp", ref_buffer_.data(), image_info_.plane_info[0].width,
            ref_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width), image_info_.plane_info[0].width,
            ref_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width) * 2, image_info_.plane_info[0].width,
            image_info_.plane_info[0].width, image_info_.plane_info[0].height);
    }
    ASSERT_EQ(out_buffer_.size(), ref_buffer_.size());
    ASSERT_EQ(0, memcmp(reinterpret_cast<void*>(ref_buffer_.data()), reinterpret_cast<void*>(out_buffer_.data()), out_buffer_.size()));
}

static const char* css_filenames[] = {"/jpeg2k/chroma_420/artificial_420_8b3c_dwt97CPRL.jp2",
    "/jpeg2k/chroma_420/cathedral_420_8b3c_dwt53RLCP.jp2", "/jpeg2k/chroma_420/deer_420_8b3c_dwt97RPCL.jp2",
    "/jpeg2k/chroma_420/leavesISO200_420_8b3c_dwt53PCRL.jp2", "/jpeg2k/chroma_420/leavesISO200_420_8b3c_dwt97CPRL.j2k",

    "/jpeg2k/chroma_422/artificial_422_8b3c_dwt53PCRL.jp2", "/jpeg2k/chroma_422/cathedral_422_8b3c_dwt97CPRL.jp2",
    "/jpeg2k/chroma_422/deer_422_8b3c_dwt53RPCL.j2k", "/jpeg2k/chroma_422/deer_422_8b3c_dwt53RPCL.jp2",
    "/jpeg2k/chroma_422/leavesISO200_422_8b3c_dwt97PCRL.jp2"};

INSTANTIATE_TEST_SUITE_P(DECODE_CHROMA_SUBSAMPLING, NvJpeg2kExtDecoderTestSingleImage, ::testing::ValuesIn(css_filenames));

}} // namespace nvimgcdcs::test