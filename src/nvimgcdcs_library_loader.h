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

#include <stdexcept>
#include <string>

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
#include <dlfcn.h>
using nvimgcdcsLibraryHandle = void *;
inline nvimgcdcsLibraryHandle nvimgcdcsLoadLibrary(const std::string &libraryPath)
{
    nvimgcdcsLibraryHandle handle = ::dlopen(libraryPath.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
        throw std::runtime_error(std::string("Failed to load library"));
    }
    return handle;
}

inline void nvimgcdcsUnloadLibrary(nvimgcdcsLibraryHandle libraryHandle)
{
    const int result = ::dlclose(libraryHandle);
    if (result != 0) {
        throw std::runtime_error(std::string("Failed to unload library"));
    }
}

template <typename FuncSignature>
inline FuncSignature *nvimgcdcsGetFuncAddress(nvimgcdcsLibraryHandle libraryHandle,
                                              const std::string &funcName)
{
    void *funcPtr = ::dlsym(libraryHandle, funcName.c_str());
    if (funcPtr == nullptr) {
        throw std::runtime_error(std::string("Failed to get function from library"));
    }
    return reinterpret_cast<FuncSignature *>(funcPtr);
}
#elif defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
using nvimgcdcsLibraryHandle = HMODULE;
inline nvimgcdcsLibraryHandle nvimgcdcsLoadLibrary(const std::string &libraryPath)
{
    nvimgcdcsLibraryHandle handle = ::LoadLibrary(libraryPath.c_str());
    if (handle == nullptr) {
        throw std::runtime_error(std::string("Failed to load library"));
    }
    return handle;
}

inline void nvimgcdcsUnloadLibrary(nvimgcdcsLibraryHandle libraryHandle)
{
    const BOOL result = ::FreeLibrary(libraryHandle);
    if (result == 0) {
        throw std::runtime_error(std::string("Failed to unload library"));
    }
}

template <typename FuncSignature>
inline FuncSignature *nvimgcdcsGetFuncAddress(nvimgcdcsModuleHandle libraryHandle,
                                              const std::string &funcName)
{
    FARPROC funcPtr = ::GetProcAddress(libraryHandle, funcName.c_str());
    if (funcPtr == nullptr) {
        throw std::runtime_error(std::string("Failed to get function from library"));
    }
    return reinterpret_cast<FuncSignature *>(funcPtr);
}
#elif defined(__APPLE__)

#endif
