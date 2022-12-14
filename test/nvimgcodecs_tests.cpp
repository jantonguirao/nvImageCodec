/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);

    cudaDeviceProp props;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);
    std::cout<<"\nUsing GPU - "<<props.name<<" with CC "<<props.major<<"."<<props.minor<<std::endl;
    int result = RUN_ALL_TESTS();
    cudaDeviceReset();
    return result;
}