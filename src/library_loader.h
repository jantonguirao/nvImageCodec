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

#include <nvimgcodecs.h>

#include <string>
#include "ilibrary_loader.h"

namespace nvimgcdcs {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
class LibraryLoader : public ILibraryLoader
{
  public:
    LibraryHandle loadLibrary(const std::string& library_path) override
    {
      return ::dlopen(library_path.c_str(), RTLD_LAZY);
    }
    void unloadLibrary(LibraryHandle library_handle) override 
    {
        const int result = ::dlclose(library_handle);
        if (result != 0) {
            throw std::runtime_error(std::string("Failed to unload library ") + dlerror());
        }
    
    }
    void* getFuncAddress(LibraryHandle library_handle, const std::string& func_name) override
    {
        return ::dlsym(library_handle, func_name.c_str());
    }
};

#elif defined(_WIN32) || defined(_WIN64)

class LibraryLoader : public ILibraryLoader
{
  public:
    LibraryHandle loadLibrary(const std::string& library_path) override
    {
        return ::LoadLibrary(library_path.c_str());
    }
    void unloadLibrary(LibraryHandle library_handle) override
    {
        const BOOL result = ::FreeLibrary(library_handle);
        if (result == 0) {
            throw std::runtime_error(std::string("Failed to unload library ") + dlerror());
        }
    }
    void* getFuncAddress(LibraryHandle library_handle, const std::string& func_name) override
    {
        return reinterpret_cast<void*>( ::GetProcAddress(library_handle, func_name.c_str()));
    }
};

#endif

} // namespace nvimgcdcs
