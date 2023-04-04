/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <type_traits>
#include <nvimgcodecs.h>
#include <cstring>
#include <cassert>
#include <stdexcept>

namespace nvimgcdcs {

namespace detail {

template <int nbytes, bool is_little_endian, typename T>
std::enable_if_t<std::is_integral<T>::value> ReadValueImpl(T &value, const uint8_t* data) {
  static_assert(sizeof(T) >= nbytes, "T can't hold the requested number of bytes");
  value = 0;
  constexpr unsigned pad = (sizeof(T) - nbytes) * 8;  // handle sign when nbytes < sizeof(T)
  for (int i = 0; i < nbytes; i++) {
    unsigned shift = is_little_endian ? (i*8) + pad: (sizeof(T)-1-i)*8;
    value |= data[i] << shift;
  }
  value >>= pad;
}

template <int nbytes, bool is_little_endian, typename T>
std::enable_if_t<std::is_enum<T>::value> ReadValueImpl(T &value, const uint8_t* data) {
  using U = std::underlying_type_t<T>;
  static_assert(nbytes <= sizeof(U),
    "`nbytes` should not exceed the size of the underlying type of the enum");
  U tmp;
  ReadValueImpl<nbytes, is_little_endian>(tmp, data);
  value = static_cast<T>(tmp);
}

template <int nbytes, bool is_little_endian>
void ReadValueImpl(float &value, const uint8_t* data) {
  static_assert(nbytes == sizeof(float),
    "nbytes is expected to be the same as sizeof(float)");
  uint32_t tmp;
  ReadValueImpl<nbytes, is_little_endian>(tmp, data);
  memcpy(&value, &tmp, sizeof(float));
}

template <int nbytes, bool is_little_endian, typename T>
void ReadValueImpl(T &value, nvimgcdcsIoStreamDesc_t io_stream) {
  uint8_t data[nbytes];  // NOLINT [runtime/arrays]
  size_t read_nbytes = 0;
  io_stream->read(io_stream->instance, &read_nbytes, data, nbytes);
  if (read_nbytes != nbytes) {
      throw std::runtime_error("Unexpected end of stream");
  }
  return ReadValueImpl<nbytes, is_little_endian>(value, data);
}

}  // namespace detail


/**
 * @brief Reads value of size `nbytes` from a stream of bytes (little-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueLE(const uint8_t* data) {
  T ret;
  detail::ReadValueImpl<nbytes, true>(ret, data);
  return ret;
}

/**
 * @brief Reads value of size `nbytes` from a stream of bytes (big-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueBE(const uint8_t* data) {
  T ret;
  detail::ReadValueImpl<nbytes, false>(ret, data);
  return ret;
}

/**
 * @brief Reads value of size `nbytes` from an input stream (little-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueLE(nvimgcdcsIoStreamDesc_t stream) {
  T ret;
  detail::ReadValueImpl<nbytes, true>(ret, stream);
  return ret;
}

/**
 * @brief Reads value of size `nbytes` from an input stream (big-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueBE(nvimgcdcsIoStreamDesc_t stream) {
  T ret;
  detail::ReadValueImpl<nbytes, false>(ret, stream);
  return ret;
}

template <typename T>
T ReadValue(nvimgcdcsIoStreamDesc_t io_stream) {
    size_t read_nbytes = 0;
    T data;
    if (NVIMGCDCS_STATUS_SUCCESS != io_stream->read(io_stream->instance, &read_nbytes, &data, sizeof(T)) || read_nbytes != sizeof(T))
        throw std::runtime_error("Failed to read");
    return data;
}

}  // namespace nvimgcdcs