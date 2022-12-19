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

#include <nvimgcdcs_module.h>
#include <memory>
#include <string>

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
    #include <dlfcn.h>
#elif defined(_WIN32) || defined(_WIN64)
    #include <Windows.h>
#endif

namespace nvimgcdcs {

class ILibraryLoader
{
  public:
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
    using LibraryHandle = void*;
#elif defined(_WIN32) || defined(_WIN64)
    using LibraryHandle = HMODULE;
#endif
    virtual ~ILibraryLoader() = default;

    virtual LibraryHandle loadLibrary(const std::string& library_path) = 0;
    virtual void unloadLibrary(LibraryHandle library_handle)           = 0;
    virtual void* getFuncAddress(
        LibraryHandle library_handle, const std::string& func_name) = 0;
};

} // namespace nvimgcdcs