// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <stdio.h>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include "library_loader.h"

namespace {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
  static const char __Nvjpeg2kLibName[] = "libnvjpeg2k.so";
#elif defined(_WIN32) || defined(_WIN64)
  static const char __Nvjpeg2kLibName[] = "libnvjpeg2k.dll";
#endif


nvimgcdcs::ILibraryLoader::LibraryHandle loadNvjpeg2kLibrary()
{
    nvimgcdcs::LibraryLoader lib_loader;
    nvimgcdcs::ILibraryLoader::LibraryHandle ret = nullptr;
    ret = lib_loader.loadLibrary(__Nvjpeg2kLibName);
    if (!ret) {
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
        fprintf(stderr, "dlopen libnvjpeg.so failed!. Please install CUDA toolkit or nvJPEG python wheel.");
#elif defined(_WIN32) || defined(_WIN64)
        fprintf(stderr, "LoadLibrary libnvjpeg.dll failed!. Please install CUDA toolkit or nvJPEG python wheel.");
#endif
    }
    return ret;
}

}  // namespace

void *Nvjpeg2kLoadSymbol(const char *name) {
  nvimgcdcs::LibraryLoader lib_loader;
  static nvimgcdcs::ILibraryLoader::LibraryHandle nvjpeg2kDrvLib = loadNvjpeg2kLibrary();
  // check processing library, core later if symbol not found
  void *ret = nvjpeg2kDrvLib ? lib_loader.getFuncAddress(nvjpeg2kDrvLib, name) : nullptr;
  return ret;
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
