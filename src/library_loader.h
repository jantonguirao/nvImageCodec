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
        LibraryHandle handle = ::dlopen(library_path.c_str(), RTLD_LAZY);
        if (handle == nullptr) {
            throw std::runtime_error(std::string("Failed to load library"));
        }
        return handle;
    }
    void unloadLibrary(LibraryHandle library_handle) override 
    {
        const int result = ::dlclose(library_handle);
        if (result != 0) {
            throw std::runtime_error(std::string("Failed to unload library"));
        }
    
    }
    void* getFuncAddress(LibraryHandle library_handle, const std::string& func_name) override
    {
        void* func_ptr = ::dlsym(library_handle, func_name.c_str());
        if (func_ptr == nullptr) {
            throw std::runtime_error(std::string("Failed to get function from library"));
        }
        return func_ptr;
    }
};

#elif defined(_WIN32) || defined(_WIN64)

class LibraryLoader : public ILibraryLoader
{
  public:
    LibraryHandle loadLibrary(const std::string& library_path) override
    {
        LibraryHandle handle = ::LoadLibrary(library_path.c_str());
        if (handle == nullptr) {
            throw std::runtime_error(std::string("Failed to load library"));
        }
        return handle;
    }
    void unloadLibrary(LibraryHandle libraryHandle) override
    {
        const BOOL result = ::FreeLibrary(library_handle);
        if (result == 0) {
            throw std::runtime_error(std::string("Failed to unload library"));
        }
    }
    void* getFuncAddress(LibraryHandle library_handle, const std::string& func_name) override
    {
        FARPROC func_ptr = ::GetProcAddress(library_handle, func_name.c_str());
        if (funcPtr == nullptr) {
            throw std::runtime_error(std::string("Failed to get function from library"));
        }
        return reinterpret_cast<void*>(func_ptr);
    }
};

#endif

} // namespace nvimgcdcs