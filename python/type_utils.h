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

#include <nvimgcodecs.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace nvimgcdcs {

inline size_t sample_type_to_bytes_per_element(nvimgcdcsSampleDataType_t sample_type)
{
    //Shift by 8 since 8..15 bits represents type bitdepth,  then shift by 3 to convert to # bytes 
    return static_cast<unsigned int>(sample_type) >> (8 + 3);
}

inline bool is_sample_format_interleaved(nvimgcdcsSampleFormat_t sample_format)
{
    //First bit of sample format says if this is interleaved or not  
    return static_cast<int>(sample_format) % 2 == 0 ;
}

inline std::string format_str_from_type(nvimgcdcsSampleDataType_t type)
{
    switch (type) {
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT8:
        return "|i1";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8:
        return "|u1";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT16:
        return "<i2";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16:
        return "<u2";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT32:
        return "<i4";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT32:
        return "<u4";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_INT64:
        return "<i8";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_UINT64:
        return "<u8";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT16:
        return "<f2";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32:
        return "<f4";
    case NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT64:
        return "<f8";
    default:
        break;
    }
    return "";
}

inline nvimgcdcsSampleDataType_t type_from_format_str(const std::string& typestr)
{
    pybind11::ssize_t itemsize = py::dtype(typestr).itemsize();
    if (itemsize == 1) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT8;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
    } else if (itemsize == 2) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT16;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT16;
    } else if (itemsize == 4) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT32;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT32;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32;
    } else if (itemsize == 8) {
        if (py::dtype(typestr).kind() == 'i')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_INT64;
        if (py::dtype(typestr).kind() == 'u')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_UINT64;
        if (py::dtype(typestr).kind() == 'f')
            return NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT64;
    }
    return NVIMGCDCS_SAMPLE_DATA_TYPE_UNKNOWN;
}

} // namespace nvimgcdcs