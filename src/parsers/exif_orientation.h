
/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodec.h>
#include <vector>

namespace nvimgcodec {

enum class ExifOrientation : uint16_t {
  HORIZONTAL = 1,
  MIRROR_HORIZONTAL = 2,
  ROTATE_180 = 3,
  MIRROR_VERTICAL = 4,
  MIRROR_HORIZONTAL_ROTATE_270_CW = 5,
  ROTATE_90_CW = 6,
  MIRROR_HORIZONTAL_ROTATE_90_CW = 7,
  ROTATE_270_CW = 8
};

inline nvimgcodecOrientation_t FromExifOrientation(ExifOrientation exif_orientation) {
  switch (exif_orientation) {
    case ExifOrientation::HORIZONTAL:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
    case ExifOrientation::MIRROR_HORIZONTAL:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, true, false};
    case ExifOrientation::ROTATE_180:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, nullptr, 180, false, false};
    case ExifOrientation::MIRROR_VERTICAL:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, true};
    case ExifOrientation::MIRROR_HORIZONTAL_ROTATE_270_CW:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, nullptr, 90, false, true};  // 270 CW = 90 CCW
    case ExifOrientation::ROTATE_90_CW:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, nullptr, 270, false, false};  // 90 CW = 270 CCW
    case ExifOrientation::MIRROR_HORIZONTAL_ROTATE_90_CW:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, nullptr, 270, false, true};  // 90 CW = 270 CCW
    case ExifOrientation::ROTATE_270_CW:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, nullptr, 90, false, false};  // 270 CW = 90 CCW
    default:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
  }
}

}  // namespace nvimgcodec