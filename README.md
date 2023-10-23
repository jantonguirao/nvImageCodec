# nvImageCodec

nvImageCodec is a library of accelerated codecs with unified interface.
It is designed as a framework for extension modules which delivers codec plugins.

Currently there are following codecs supported:
- nvjpeg
- nvjpeg2000
- nvbmp (as an example extension module)
- nvpnm (as an example extension module with encoder/writter only)
- libturbo-jpeg (decoder only)
- libtiff (decoder only)
- opencv (decoders only)

## Getting Started

To get a local copy up and running follow these steps.

### Pre-requisites
- git
- git lfs (images used for testing are stored as lfs files)
- CMake >= 3.14
- gcc >= 9.4
- NVIDIA CUDA >= 11.8
- libjpeg-turbo >= 2.0.0
- opencv >= 3.0.0
- Python for tests and examples
- Python packages: clang==14 wheel setuptools
- Supported systems:
  - Windows >= 10
  - Ubuntu >= 20.04
  - WSL2 with Ubuntu >= 20.04


### Installation

The following steps describe how to install nvImageCodec from pre-built install packages.
Choose the installation method that meets your environment needs.

#### Tar file installation

```
tar -xvf nvimgcodec-0.1.0_alpha.1-cuda12-x86_64-linux-lib.tar.gz -C /opt/nvidia/
```

#### DEB File Installation
```
sudo apt-get install -y ./nvimgcodec-0.1.0_alpha.1-cuda12-x86_64-linux-lib.deb
```
#### Python WHL File Installation

`x` in the below command is the build id. It will be 0 when the .whl is built locally.

```
pip install nvidia_nvimgcodec_cuda12-0.1.0.x-py3-none-manylinux2014_x86_64.whl
```

### Build

```
git lfs clone ssh://git@gitlab-master.nvidia.com:12051/cuda-hpc-libraries/nvimagecodec.git
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

##### Windows

To install nvImageCodec on Windows please execute following commands in console with administrator privileges
```
cd build
cmake --install . --config Release
```
All necessary files should be installed in C:\Program Files\nvimgcodec_cuda<major_cuda_ver> directory.

```
C:\Program Files\nvimgcodec_cuda<major_cuda_ver>
│   LICENSE.txt
│
├───bin
│       nvimgcodec.dll
│       nvimtrans.exe
│
├───include
│       nvimgcodec.h
│       nvimgcodec_version.h
│       nvimgcodec.h
│
├───lib64
│       nvimgcodec.lib
│       nvimgcodec_static.lib
│
├───extensions
│       nvbmp_ext_0.dll
│       nvjpeg2k_ext_0.dll
│       nvjpeg_ext_22.dll
│
└───test


```

To install in other folder please use --prefix argument
```
cmake --install . --config Release --prefix bin
```

## Testing
Run CTest to execute L0 and L2 tests
```
cd build
cmake --install . --config Release --prefix bin
ctest -C Release
```

Run L2 pytest
```
cd build
cmake --install . --config Release --prefix bin
pytest ../test
```

## Packaging

From a successfully built project, installers can be generated using cpack:
```
cd build
cpack --config CPackConfig.cmake -DCMAKE_BUILD_TYPE=Release
```
This will generate in build directory *.zip or *tar.xz files

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

## Requirements
- [nvImageCodec PRD](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdocs.google.com%2Fdocument%2Fd%2F1KrFzidHNfozNYk8a3crs0ekNH3ETisT1%2Fedit&data=05%7C01%7Csmatysik%40nvidia.com%7C7a7093b7b5804d1b98f008dac16b827e%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638034964732398522%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=GD26jloLP4IdjvI%2BdYrmIs5PZgYCMHXWMXnLjGRfAJ4%3D&reserved=0)

## Design
- [Design slides](https://nam11.safelinks.protection.outlook.com/ap/p-59584e83/?url=https%3A%2F%2Fnvidia-my.sharepoint.com%2F%3Ap%3A%2Fp%2Ftrybicki%2FEbDMoASyk0hLukzPdpW66S4BzOvJZ9vymm0fkddy7utfkw%3Fe%3DMlduBI&data=05%7C01%7Csmatysik%40nvidia.com%7C347ebe243c764d22761908dad7cbbad2%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638059567268905928%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=xut9HNCGgftyfTR635%2BJu2Amp%2F6bF2eZsjkzhrpNOYg%3D&reserved=0)
- [Design recording](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fnvidia-my.sharepoint.com%2F%3Av%3A%2Fp%2Ftrybicki%2FEeC0aKfe5bdFixtDmg7J3ZkBJg3Pzyl1RfPkNFyQOV2VFQ&data=05%7C01%7Csmatysik%40nvidia.com%7C347ebe243c764d22761908dad7cbbad2%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638059567269062080%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=WmjhZpi1SocpDVAP5QtcM4kOQ6aiW%2FspDvMYPGwzXbQ%3D&reserved=0)

## Coding Style Guide

- [CUDA/C++ Coding Style Guide](https://docs.google.com/document/d/1jNvQBMQhoIQMSot4WFUop8Bl2bCUxvuX7Xa4910RDQI/edit)
- There is .clang-format file in the main project directory
