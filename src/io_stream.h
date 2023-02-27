/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cstdio>
#include <memory>
#include <string>
#include <stdexcept>
#include <cassert>

namespace nvimgcdcs {

/**
 * @brief An abstract file-like interface for reading and writting data.
 */
class IoStream
{
  public:
    virtual ~IoStream() = default;

    /**
     * @brief Reads all requested data from the stream; if not all of the data can be read,
     *        an exception is thrown.
     *
     * @param buf   the output buffer
     * @param bytes the number of bytes to read
     */
    inline void readBytes(void* buf, size_t bytes)
    {
        char* b = static_cast<char*>(buf);
        while (bytes) {
            int64_t n = read(b, bytes);
            if (n == 0)
                throw std::runtime_error("End of stream");
            if (n < 0)
                throw std::runtime_error("An error occurred while reading data.");
            b += n;
            assert(static_cast<size_t>(n) <= bytes);
            bytes -= n;
        }
    }

    /**
     * @brief Reads one object of given type from the stream
     *
     * @tparam T  the type of the object to read; should be trivially copyable or otherwise
     *            safe to be overwritten with memcpy or similar.
     */
    template <typename T>
    inline T readOne()
    {
        T t;
        readAll(&t, 1);
        return t;
    }

    /**
     * @brief Reads `count` instances of type T from the stream to the provided buffer
     *
     * If the function cannot read the requested number of objects, an exception is thrown
     *
     * @tparam T    the type of the object to read; should be trivially copyable or otherwise
     *              safe to be overwritten with memcpy or similar.
     * @param buf   the output buffer
     * @param count the number of objects to read
     */
    template <typename T>
    inline void readAll(T* buf, size_t count)
    {
        readBytes(buf, sizeof(T) * count);
    }

    /**
     * @brief Skips `count` objects in the stream
     *
     * Skips over the length of `count` objects of given type (by default char,
     * because sizeof(char) == 1).
     *
     * NOTE: Negative skips are allowed.
     *
     * @tparam T type of the object to skip; defaults to `char`
     * @param count the number of objects to skip
     */
    template <typename T = char>
    void skip(int64_t count = 1)
    {
        seek(count * sizeof(T), SEEK_CUR);
    }

    /**
     * @brief Reads data from the stream and advances the read pointer; partial reads are allowed.
     *
     * A valid implementation of this function reads up to `bytes` bytes from the stream and
     * stores them in `buf`. If the function cannot read all of the requested bytes due to
     * end-of-file, it shall read all it can and return the number of bytes actually read.
     * When reading from a regular file and the file pointer is already at the end, the function
     * shall return 0.
     *
     * This function does not throw EndOfStream.
     *
     * @param buf       the output buffer
     * @param bytes     the number of bytes to read
     * @return size _t  the number of bytes actually read or
     *                  0 in case of end-of-stream condition
     */
    virtual std::size_t read(void* buf, std::size_t bytes) = 0;
    virtual std::size_t write(void* buf, std::size_t bytes) = 0;
    virtual std::size_t putc(unsigned char buf) = 0;

    /**
     * @brief Moves the read pointer in the stream.
     *
     * If the new pointer would be out of range, the function may either clamp it to a valid range
     * or throw an error.
     *
     * @param pos     the offset to move
     * @param whence  the beginning - SEEK_SET, SEEK_CUR or SEEK_END
     */
    virtual void seek(int64_t pos, int whence = SEEK_SET) = 0;

    /**
     * @brief Returns the current read position, in bytes from the beginnging, in the stream.
     */
    virtual int64_t tell() const = 0;

    /**
     * @brief Returns the length, in bytes, of the stream.
     */
    virtual std::size_t size() const = 0;

    /**
     * @brief Returns the size as a signed integer.
     */
    inline int64_t ssize() const { return size(); }
};

} // namespace nvimgcdcs