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

#include <fstream>
#include <string>
#include <vector>

namespace nvimgcodec {
namespace test {

inline std::vector<uint8_t> read_file(const std::string &filename) {
  std::ifstream stream(filename, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

inline std::vector<uint8_t> replace(const std::vector<uint8_t>& data, const std::vector<uint8_t>& old_value, const std::vector<uint8_t>& new_value)
{
    std::vector<uint8_t> result;
    result.reserve(data.size());
    auto it = data.begin();
    size_t n = old_value.size();
    while (it != data.end()) {
        if (it + n <= data.end() && std::equal(it, it + n, old_value.begin(), old_value.end())) {
            result.insert(result.end(), new_value.begin(), new_value.end());
            it += n;
        } else {
            result.push_back(*(it++));
        }
    }
    return result;
}

inline void expect_eq(nvimgcodecImageInfo_t expected, nvimgcodecImageInfo_t actual) {
    EXPECT_EQ(expected.type, actual.type);
    EXPECT_EQ(expected.next, actual.next);
    EXPECT_EQ(expected.sample_format, actual.sample_format);
    EXPECT_EQ(expected.num_planes, actual.num_planes);
    EXPECT_EQ(expected.color_spec, actual.color_spec);
    EXPECT_EQ(expected.chroma_subsampling, actual.chroma_subsampling);
    EXPECT_EQ(expected.orientation.rotated, actual.orientation.rotated);
    EXPECT_EQ(expected.orientation.flip_x, actual.orientation.flip_x);
    EXPECT_EQ(expected.orientation.flip_y, actual.orientation.flip_y);
    for (int p = 0; p < expected.num_planes; p++) {
        EXPECT_EQ(expected.plane_info[p].height, actual.plane_info[p].height);
        EXPECT_EQ(expected.plane_info[p].width, actual.plane_info[p].width);
        EXPECT_EQ(expected.plane_info[p].num_channels, actual.plane_info[p].num_channels);
        EXPECT_EQ(expected.plane_info[p].sample_type, actual.plane_info[p].sample_type);
        EXPECT_EQ(expected.plane_info[p].precision, actual.plane_info[p].precision);
    }
}

inline void LoadImageFromFilename(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t& stream_handle, const std::string& filename)
{
    if (stream_handle) {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle));
        stream_handle = nullptr;
    }
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromFile(instance, &stream_handle, filename.c_str()));
}

inline void LoadImageFromHostMemory(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t& stream_handle, const uint8_t* data, size_t data_size)
{
    if (stream_handle) {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle));
        stream_handle = nullptr;
    }
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromHostMem(instance, &stream_handle, data, data_size));
}

}  // namespace test
}  // namespace nvimgcodec