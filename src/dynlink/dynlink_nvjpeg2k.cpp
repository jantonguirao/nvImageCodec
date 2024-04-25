/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda.h>
#include <stdio.h>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include "library_loader.h"
#include <nvjpeg2k_version.h>
#include <iostream>
#define STR_IMPL_(x) #x      //stringify argument
#define STR(x) STR_IMPL_(x)  //indirection to expand argument macros

namespace {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
  static const char* __Nvjpeg2kLibNames[] = {
    "libnvjpeg2k.so." STR(NVJPEG2K_VER_MAJOR),
    "libnvjpeg2k.so"
  };
#elif defined(_WIN32) || defined(_WIN64)
  static const char* __Nvjpeg2kLibNames[] = {
    "nvjpeg2k_" STR(NVJPEG2K_VER_MAJOR) ".dll"
    "nvjpeg2k.dll"
  };
#endif


nvimgcodec::ILibraryLoader::LibraryHandle loadNvjpeg2kLibrary()
{
    nvimgcodec::LibraryLoader lib_loader;
    nvimgcodec::ILibraryLoader::LibraryHandle ret = nullptr;
    for (const char* libname : __Nvjpeg2kLibNames) {
        ret = lib_loader.loadLibrary(libname);
        if (ret != nullptr)
            break;
    }
    if (!ret) {
        fprintf(stderr,
            "Failed to load nvjpeg2k library! Please install nvJPEG2000 (see "
            "https://docs.nvidia.com/cuda/nvjpeg2000/userguide.html#installing-nvjpeg2000).\n"
            "Note: If using nvImageCodec's Python distribution, "
            "it is enough to install the nvJPEG2000 wheel: e.g. `python3 -m pip install nvidia-nvjpeg2k-cu" STR(CUDA_VERSION_MAJOR) "`\n");
    }
    return ret;
}
}  // namespace

void *Nvjpeg2kLoadSymbol(const char *name) {
  nvimgcodec::LibraryLoader lib_loader;
  static nvimgcodec::ILibraryLoader::LibraryHandle nvjpeg2kDrvLib = loadNvjpeg2kLibrary();
  // check processing library, core later if symbol not found
  try {
    void *ret = nvjpeg2kDrvLib ? lib_loader.getFuncAddress(nvjpeg2kDrvLib, name) : nullptr;
    return ret;
  } catch (...) {
    return nullptr;
  }
}

bool nvjpeg2kIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = Nvjpeg2kLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
