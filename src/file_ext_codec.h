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

#include <map>
#include <string>

namespace nvimgcdcs {

 std::string file_ext_to_codec(const std::string& file_ext)
 {
     static std::map<std::string, std::string> ext2codec = {{".bmp", "bmp"}, {".j2c", "jpeg2k"}, {".j2k", "jpeg2k"}, {".jp2", "jpeg2k"},
         {".tiff", "tiff"}, {".tif", "tiff"}, {".jpg", "jpeg"}, {".jpeg", "jpeg"}, {".ppm", "pnm"}, {".pgm", "pnm"}, {".pbm", "pnm"}};
     std::string codec_name{};
     auto it = ext2codec.find(file_ext);
     if (it != ext2codec.end()) {
         codec_name = it->second;
     }
     return codec_name;
 }

}
