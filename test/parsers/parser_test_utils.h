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

namespace nvimgcdcs {
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

}  // namespace test
}  // namespace nvimgcdcs