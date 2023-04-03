/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "nvimgcodecs_tests.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>

namespace nvimgcdcs { namespace test {

std::string resources_dir;

}} // namespace nvimgcdcs::test

namespace {
std::string getCmdOption(char** begin, char** end, const std::string& option)
{
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return {};
}
} // namespace

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);

    nvimgcdcs::test::resources_dir = getCmdOption(argv, argv + argc, "--resources_dir");
    if (nvimgcdcs::test::resources_dir.empty()) {
        std::cerr << "Need a valid resources dir (e.g. --resources_dir path/to/resources)\n";
        return 1;
    }

    cudaDeviceProp props;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);
    std::cout<<"\nUsing GPU - "<<props.name<<" with CC "<<props.major<<"."<<props.minor<<std::endl;
    int result = RUN_ALL_TESTS();
    cudaDeviceReset();
    return result;
}