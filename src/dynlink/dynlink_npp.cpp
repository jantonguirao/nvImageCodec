// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include "library_loader.h"

namespace {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
  static const char __NppcLibName[] = "libnppc.so";
  static const char __NppideiLibName[] = "libnppidei.so";

  #if CUDA_VERSION >= 12000
    static const char __NppcLibName[] = "libnppc.so.12";
    static const char __NppideiLibName[] = "libnppidei.so.12";
  #elif CUDA_VERSION >= 11000 && CUDA_VERSION < 12000
    static const char __NppcLibNameCuVer[] = "libnppc.so.11";
    static const char __NppideiLibNameCuVer[] = "libnppidei.so.11";
  #else
    static const char __NppcLibNameCuVer[] = "libnppc.so.10";
    static const char __NppideiLibNameCuVer[] = "libnppidei.so.10";
  #endif

#elif defined(_WIN32) || defined(_WIN64)
  static const char __NppcLibName[] = "nppc.dll";
  static const char __NppideiLibName[] = "nppidei.dll";

  #if CUDA_VERSION >= 12000
    static const char __NppcLibNameCuVer[] = "nppc64_12.dll";
    static const char __NppideiLibNameCuVer[] = "nppidei64_12.dll";
  #elif CUDA_VERSION >= 11000 && CUDA_VERSION < 12000
    static const char __NppcLibNameCuVer[] = "nppc64_11.dll";
    static const char __NppideiLibNameCuVer[] = "nppidei64_11.dll";
  #else
    static const char __NppcLibNameCuVer[] = "nppc64_10.dll";
    static const char __NppideiLibNameCuVer[] = "nppidei64_10.dll";
  #endif
#endif

nvimgcdcs::ILibraryLoader::LibraryHandle loadNppcLibrary()
{
    nvimgcdcs::LibraryLoader lib_loader;
    nvimgcdcs::ILibraryLoader::LibraryHandle ret = nullptr;
    ret = lib_loader.loadLibrary(__NppcLibNameCuVer);
    if (!ret) {
        ret = lib_loader.loadLibrary(__NppcLibName);
        if (!ret) {
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
            fprintf(stderr, "dlopen libnppc.so failed!. Please install CUDA toolkit or NPP python wheel.");
#elif defined(_WIN32) || defined(_WIN64)
            fprintf(stderr, "LoadLibrary nppc.dll failed!. Please install CUDA toolkit or NPP python wheel.");
#endif
        }
    }
    return ret;
}

nvimgcdcs::ILibraryLoader::LibraryHandle loadNppideiLibrary()
{
    nvimgcdcs::LibraryLoader lib_loader;
    nvimgcdcs::ILibraryLoader::LibraryHandle ret = nullptr;
    ret = lib_loader.loadLibrary(__NppideiLibNameCuVer);
    if (!ret) {
        ret = lib_loader.loadLibrary(__NppideiLibName);
        if (!ret) {
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
            fprintf(stderr, "dlopen libnppidei.so failed!. Please install CUDA toolkit or NPP python wheel.");
#elif defined(_WIN32) || defined(_WIN64)
            fprintf(stderr, "LoadLibrary libnppidei.dll failed!. Please install CUDA toolkit or NPP python wheel.");
#endif
        }
    }
    return ret;
}

}  // namespace

// Load a symbol from all NPP libs that we use, to provide a unified interface
void *NppLoadSymbol(const char *name) {
  nvimgcdcs::LibraryLoader lib_loader;
  static nvimgcdcs::ILibraryLoader::LibraryHandle nppcDrvLib = loadNppcLibrary();
  static nvimgcdcs::ILibraryLoader::LibraryHandle nppideiDrvLib = loadNppideiLibrary();
  // check processing library, core later if symbol not found
  void *ret = nppideiDrvLib ? lib_loader.getFuncAddress(nppideiDrvLib, name) : nullptr;
  if (!ret) {
    ret = nppcDrvLib ? lib_loader.getFuncAddress(nppcDrvLib, name) : nullptr;
  }
  return ret;
}

bool nppIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = NppLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
