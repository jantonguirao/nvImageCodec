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

#include "input_stream.h"

namespace nvimgcdcs {

class FileInputStream : public InputStream
{
  public:
    class MappingReserver
    {
      public:
        explicit MappingReserver(unsigned int num)
            : reserved(0)
        {
            if (FileInputStream::reserveFileMappings(num)) {
                reserved = num;
            }
        }

        MappingReserver()
            : MappingReserver(0)
        {
        }

        MappingReserver(const MappingReserver&)            = delete;
        MappingReserver& operator=(const MappingReserver&) = delete;

        MappingReserver(MappingReserver&& other)
            : MappingReserver(other.reserved)
        {
            other.reserved = 0;
        }

        MappingReserver& operator=(MappingReserver&& other)
        {
            reserved       = other.reserved;
            other.reserved = 0;
            return *this;
        }

        MappingReserver& operator=(MappingReserver& other)
        {
            reserved       = other.reserved;
            other.reserved = 0;
            return *this;
        }

        bool CanShareMappedData() { return reserved != 0; }

        ~MappingReserver()
        {
            if (reserved) {
                FileInputStream::freeFileMappings(reserved);
            }
        }

      private:
        unsigned int reserved;
    };

    static std::unique_ptr<FileInputStream> open(const std::string& uri, bool read_ahead, bool use_mmap);

    virtual void close()                         = 0;
    virtual std::shared_ptr<void> get(size_t n_bytes) = 0;
    virtual ~FileInputStream()                   = default;

  protected:
    static bool reserveFileMappings(unsigned int num);
    static void freeFileMappings(unsigned int num);
    explicit FileInputStream(const std::string& path)
        : path_(path)
    {
    }

    std::string path_;
};

} // namespace nvimgcdcs