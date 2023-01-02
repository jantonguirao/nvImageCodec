# nvImageCodecs

## Description

nvImageCodecs is a library of accelerated codecs with unified interface. It is designed as a framework for extension modules which delivers particular codec plugins.

## Requirements
- [nvImageCodecs PRD](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdocs.google.com%2Fdocument%2Fd%2F1KrFzidHNfozNYk8a3crs0ekNH3ETisT1%2Fedit&data=05%7C01%7Csmatysik%40nvidia.com%7C7a7093b7b5804d1b98f008dac16b827e%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638034964732398522%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=GD26jloLP4IdjvI%2BdYrmIs5PZgYCMHXWMXnLjGRfAJ4%3D&reserved=0)

## Design
- [Design slides](https://nam11.safelinks.protection.outlook.com/ap/p-59584e83/?url=https%3A%2F%2Fnvidia-my.sharepoint.com%2F%3Ap%3A%2Fp%2Ftrybicki%2FEbDMoASyk0hLukzPdpW66S4BzOvJZ9vymm0fkddy7utfkw%3Fe%3DMlduBI&data=05%7C01%7Csmatysik%40nvidia.com%7C347ebe243c764d22761908dad7cbbad2%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638059567268905928%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=xut9HNCGgftyfTR635%2BJu2Amp%2F6bF2eZsjkzhrpNOYg%3D&reserved=0)
- [Design recording](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fnvidia-my.sharepoint.com%2F%3Av%3A%2Fp%2Ftrybicki%2FEeC0aKfe5bdFixtDmg7J3ZkBJg3Pzyl1RfPkNFyQOV2VFQ&data=05%7C01%7Csmatysik%40nvidia.com%7C347ebe243c764d22761908dad7cbbad2%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638059567269062080%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=WmjhZpi1SocpDVAP5QtcM4kOQ6aiW%2FspDvMYPGwzXbQ%3D&reserved=0)

## Coding Style Guide

- [CUDA/C++ Coding Style Guide](https://docs.google.com/document/d/1jNvQBMQhoIQMSot4WFUop8Bl2bCUxvuX7Xa4910RDQI/edit)
- There is .clang-format file in the main project directory

## Prototype
Currently this project is in prototype state  with initial support for:
- nvJPEG
- nvJPEG2000
- nvBMP (as a example extension module)

There are following known limitation:
- Only planar RGB format is supported as a input for encoder and output from decoder
- There is limited set of supported decoder and encoder parameters
- No custom allocators yet
- No batch processing yet
- No ROI support yet
- No Metadata support yet (appart of EXIF orientation for jpeg)

## Pre-requisites
- git
- git lfs (images used for testing are stored as lfs files) 
- CMake >= 3.14
- gcc >= 9.4 
- NVIDIA CUDA >= 11.8 
- Python for tests and examples
- Supported systems:
  - Windows >= 10 
  - Ubuntu >= 20.04
  - WSL2 with Ubuntu >= 20.04 

## Build

```
git lfs clone ssh://git@gitlab-master.nvidia.com:12051/cuda-hpc-libraries/nvimagecodec.git
cd nvimagecodec
git checkout prototype
git submodule update --init --recursive --remote
mkdir build
cd build
export CUDACXX=nvcc
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
## Install for tests

Below is described temporar instalation process just for testing prototype 

### Linux
```
cd build
cmake --install . --config Release --prefix bin
cd bin
sudo ./install.sh
```

After execution there should be:
- all plugins in /usr/lib/nvimgcodecs/plugins (it is default directory for plugin discovery)
- libnvimgcodecs.so in /usr/lib/x86_64-linux-gnu

### Windows

To install nvImageCodecs on Windows please execute following commmands in console with administrator privileges
```
cd build
cmake --install . --config Release
```
All necessery files should be installed in C:\Program Files\nvimgcodecs directory.

```
C:\Program Files\nvimgcodecs
│   LICENSE.txt
│
├───bin
│       nvimgcodecs.dll
│       nvimgcodecs_example_high_level_api.exe
│       nvimtrans.exe
│       nvjpeg2k_example.exe
│
├───include
│       nvimgcodecs.h
│       nvimgcdcs_version.h
│       nvimgcodecs.h
│       nvjpeg.h
│       nvjpeg2k.h
│       nvjpeg2k_version.h
│
├───lib64
│       nvbmp.lib
│       nvimgcodecs.lib
│       nvimgcodecs_static.lib
│       nvjpeg.lib
│       nvjpeg2k.lib
│
├───plugins
│       nvbmp_0.dll
│       nvjpeg2k_0.dll
│       nvjpeg64_22.dll
│
├───python
│       nvimgcodecs.cp39-win_amd64.pyd│
│
└───test
        nvjpeg2k_negative_tests.exe
        nvjpeg2k_perf_tests.exe
        nvjpeg2k_tests.exe
        nvjpeg_example.exe
        nvjpeg_example_batched.exe
        nvjpeg_example_encode.exe
        nvjpeg_example_new.exe
        nvjpeg_example_transcode.exe
        nvjpeg_exposed.exe
        nvjpeg_tests.exe
        nvjpeg_tests_L2.exe
        nvjpeg_tests_perf.exe

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

From a succesfully built project, installers can be generated using cpack:
```
cd build
cpack --config CPackConfig.cmake -DCMAKE_BUILD_TYPE=Release
```
This will generate in build directory *.zip or *tar.xz files

