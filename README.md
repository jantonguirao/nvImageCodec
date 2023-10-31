# nvImageCodec

[![License](https://img.shields.io/badge/License-Apache_2.0-yellogreen.svg)](https://opensource.org/licenses/Apache-2.0)

![Version](https://img.shields.io/badge/Version-v0.2.0--beta-blue)

![Platform](https://img.shields.io/badge/Platform-linux--64_%7C_win--64_wsl2-gray)

[![Cuda](https://img.shields.io/badge/CUDA-v11.8-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)
[![GCC](https://img.shields.io/badge/GCC-v9.4-yellow)](https://gcc.gnu.org/gcc-9/)
[![Python](https://img.shields.io/badge/python-v3.8_%7c_v3.9_%7c_v3.10%7c_v3.11%7c_v3.12-blue?logo=python)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-v3.24-%23008FBA?logo=cmake)](https://cmake.org/)
                                                                                                        

The nvImageCodec is an open-source library of accelerated codecs with unified interface. It is designed as a framework for extension modules which delivers codec plugins.

This nvImageCodec release includes the following key features:

- Unified API for decoding and encoding images
- Batch processing, with variable shape and heterogeneous formats images
- Codec prioritization with automatic fallback
- Builtin parsers for image format detection: jpeg, jpeg2000, tiff, bmp, png, pnm, webp 
- Python bindings
- Zero-copy interfaces to CV-CUDA, PyTorch and CuPy 
- End-end accelerated sample applications for common image transcoding

Currently there are following native codec extensions:

- nvjpeg_ext

   - Hardware jpeg decoder
   - CUDA jpeg decoder
   - CUDA lossless jpeg decoder
   - CUDA jpeg encoder

- nvjpeg2k_ext

   - CUDA jpeg 2000 decoder (including High Throughput Jpeg2000)
   - CUDA jpeg 2000 encoder 

- nvbmp_ext (as an example extension module)

   - CPU bmp reader
   - CPU bmp writer

- nvpnm_ext (as an example extension module)

   - CPU pnm (ppm, pbm, pgm) writer

Additionally as a fallback there are following 3rd party codec extensions:

- libturbo-jpeg_ext

   - CPU jpeg decoder

- libtiff_ext 

   - CPU tiff decoder

- opencv_ext

   - CPU jpeg decoder
   - CPU jpeg2k_decoder
   - CPU png decoder
   - CPU bmp decoder
   - CPU pnm decoder
   - CPU tiff decoder
   - CPU webp decoder


## Getting Started

To get a local copy up and running follow these steps.

### Pre-requisites

-  Linux distro:
   - Ubuntu x86_64 >= 20.04
   - WSL2 with Ubuntu >= 20.04 (tested with 20.04)
- NVIDIA driver >= 520.56.06
- CUDA Toolkit > = 11.8
- GCC >= 9.4
- Python >= 3.8
- cmake >= 3.24

### Installation

The following steps describe how to install nvImageCodec from pre-built install packages.
Choose the installation method that meets your environment needs. The `x` letter in the below command is the build id. 
It will be 0 when the package is built locally.

#### Tar file installation

```
tar -xvf nvimgcodec-0.2.0.x_beta-cuda12-x86_64-linux-lib.tar.gz -C /opt/nvidia/
```

#### DEB File Installation
```
sudo apt-get install -y ./nvimgcodec-0.2.0.x_beta-cuda12-x86_64-linux-lib.deb
```
#### Python WHL File Installation

```
pip install nvidia_nvimgcodec_cu12-0.2.0.x_beta-py3-none-manylinux2014_x86_64.whl
```

## Build and install from Sources

### Additional pre-requisites
- git
- git lfs (images used for testing are stored as lfs files)
- patchelf >= 0.17.2
- Dependencies for extensions. If you would not like to build particular extension you can skip it.
  - nvJPEG2000 >= 0.7.0
  - libjpeg-turbo >= 2.0.0
  - libtiff >= 4.5.0
  - opencv >= 4.7.0
- Python packages: 
  - clang==14.0.1 
  - wheel
  - setuptools
  - sphinx_rtd_theme
  - breathe 
  - future
  - flake8
  - sphinx==4.5.0

Please see also Dockerfiles.

### Build

```
git lfs clone https://github.com/NVIDIA/nvImageCodec.git
cd nvimagecodec
git submodule update --init --recursive --remote
mkdir build
cd build
export CUDACXX=nvcc
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

#### Build CVCUDA samples

To build CV-CUDA samples, additionally CV-CUDA has to be installed and CVCUDA_DIR and NVCV_TYPES_DIR
need to point folders with *-config.cmake files. Apart of that, BUILD_CVCUDA_SAMPLES variable must be set to ON.


## Build Python wheel

After succesfully built project, execute below commands.

```
cd build
make wheel
```

## Packaging

From a successfully built project, installers can be generated using cpack:
```
cd build
cpack --config CPackConfig.cmake -DCMAKE_BUILD_TYPE=Release
```
This will generate in build directory *.zip or *tar.xz files

### Installation from sources

##### Linux
```
cd build
cmake --install . --config Release -prefix /opt/nvidia/nvimgcodec_<major_cuda_ver>
```

After execution there should be:
- all extension modules in /opt/nvidia/nvimgcodec_cuda<major_cuda_ver>/extensions (it is default directory for extension discovery)
- libnvimgcodec.so in /opt/nvidia/nvimgcodec_cuda<major_cuda_ver>/lib64

Add directory with libnvimgcodec.so to LD_LIBRARY_PATH

```
export LD_LIBRARY_PATH=/opt/nvidia/nvimgcodec_cuda<major_cuda_ver>/lib64:$LD_LIBRARY_PATH
```


## Testing
Run CTest to execute L0 and L1 tests
```
cd build
cmake --install . --config Release --prefix bin
ctest -C Release
```

Run sample transcoder app tests
```
cd build
cmake --install . --config Release --prefix bin
cd bin/test

LD_LIBRARY_PATH=$PWD/../lib64 pytest -v test_transcoder.py

```

Run Python API tests

First install python wheel. You would also need to have installed all Python tests dependencies (see Dockerfiles). 

```
pip install nvidia_nvimgcodec_cu12-0.2.0.x_beta-py3-none-manylinux2014_x86_64.whl
```

Run tests
```
cd tests
pytest -v ./python
```

## CMake package integration

To use nvimagecodec as a dependency in your CMake project, use:
```
list(APPEND CMAKE_PREFIX_PATH "/opt/nvidia/nvimgcodec_cuda<major_cuda_ver>/")  # or the prefix where the package was installed if custom

find_package(nvimgcodec CONFIG REQUIRED)
# Mostly for showing some of the variables defined
message(STATUS "nvimgcodec_FOUND=${nvimgcodec_FOUND}")
message(STATUS "nvimgcodec_INCLUDE_DIR=${nvimgcodec_INCLUDE_DIR}")
message(STATUS "nvimgcodec_LIB_DIR=${nvimgcodec_LIB_DIR}")
message(STATUS "nvimgcodec_BIN_DIR=${nvimgcodec_BIN_DIR}")
message(STATUS "nvimgcodec_LIB=${nvimgcodec_LIB}")
message(STATUS "nvimgcodec_EXTENSIONS_DIR=${nvimgcodec_EXTENSIONS_DIR}")
message(STATUS "nvimgcodec_VERSION=${nvimgcodec_VERSION}")

target_include_directories(<your-target> PUBLIC ${nvimgcodec_INCLUDE_DIR})
target_link_directories(<your-target> PUBLIC ${nvimgcodec_LIB_DIR})
target_link_libraries(<your-target> PUBLIC ${nvimgcodec_LIB})
```

