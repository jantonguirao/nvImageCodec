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
using nvimgcdcsModuleHandle = void *;
inline nvimgcdcsModuleHandle nvimgcdcsLoadModule(const std::string &modulePath)
{
    nvimgcdcsModuleHandle handle = ::dlopen(modulePath.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
        throw std::runtime_error(std::string("Failed to load module"));
    }
    return handle;
}

inline void nvimgcdcsUnloadModule(nvimgcdcsModuleHandle moduleHandle)
{
    const int result = ::dlclose(moduleHandle);
    if (result == 0) {
        throw std::runtime_error(std::string("Failed to unload module"));
    }
}

template <typename FuncSignature>
inline FuncSignature *nvimgcdcsGetFuncAddress(nvimgcdcsModuleHandle moduleHandle,
                                              const std::string &funcName)
{
    void *funcPtr = ::dlsym(moduleHandle, funcName.c_str());
    if (funcPtr == nullptr) {
        throw std::runtime_error(std::string("Failed to get function from module"));
    }
    return reinterpret_cast<FuncSignature *>(funcPtr);
}
#elif defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
using nvimgcdcsModuleHandle = HMODULE;
inline nvimgcdcsModuleHandle nvimgcdcsLoadModule(const std::string &modulePath)
{
    nvimgcdcsModuleHandle handle = ::LoadLibrary(modulePath.c_str());
    if (handle == nullptr) {
        throw std::runtime_error(std::string("Failed to load module"));
    }
    return handle;
}

inline void nvimgcdcsUnloadModule(nvimgcdcsModuleHandle moduleHandle)
{
    const BOOL result = ::FreeLibrary(moduleHandle);
    if (result == 0) {
        throw std::runtime_error(std::string("Failed to unload module"));
    }
}

template <typename FuncSignature>
inline FuncSignature *nvimgcdcsGetFuncAddress(nvimgcdcsModuleHandle moduleHandle,
                                              const std::string &funcName)
{
    FARPROC funcPtr = ::GetProcAddress(moduleHandle, funcName.c_str());
    if (funcPtr == nullptr) {
        throw std::runtime_error(std::string("Failed to get function from module"));
    }
    return reinterpret_cast<FuncSignature *>(funcPtr);
}
#elif defined(__APPLE__)

#endif
