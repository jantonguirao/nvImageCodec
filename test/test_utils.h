
/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

namespace nvimgcdcs { namespace test {

int write_bmp(const char* filename, const unsigned char* d_chanR, int pitchR, const unsigned char* d_chanG, int pitchG,
    const unsigned char* d_chanB, int pitchB, int width, int height);

}} // namespace nvimgcdcs::test
