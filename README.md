# nvImageCodecs

## Description

nvImageCodecs is a library of accelerated codecs with unified interface. It is designed as a framework for extension modules which delivers particular codec plugins.

## Requirements
- [ ] [nvImageCodecs PRD](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdocs.google.com%2Fdocument%2Fd%2F1KrFzidHNfozNYk8a3crs0ekNH3ETisT1%2Fedit&data=05%7C01%7Csmatysik%40nvidia.com%7C7a7093b7b5804d1b98f008dac16b827e%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638034964732398522%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=GD26jloLP4IdjvI%2BdYrmIs5PZgYCMHXWMXnLjGRfAJ4%3D&reserved=0)

## Design
- [ ] [Design slides](https://nam11.safelinks.protection.outlook.com/ap/p-59584e83/?url=https%3A%2F%2Fnvidia-my.sharepoint.com%2F%3Ap%3A%2Fp%2Ftrybicki%2FEbDMoASyk0hLukzPdpW66S4BzOvJZ9vymm0fkddy7utfkw%3Fe%3DMlduBI&data=05%7C01%7Csmatysik%40nvidia.com%7C347ebe243c764d22761908dad7cbbad2%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638059567268905928%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=xut9HNCGgftyfTR635%2BJu2Amp%2F6bF2eZsjkzhrpNOYg%3D&reserved=0)
- [ ] [Design recording](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fnvidia-my.sharepoint.com%2F%3Av%3A%2Fp%2Ftrybicki%2FEeC0aKfe5bdFixtDmg7J3ZkBJg3Pzyl1RfPkNFyQOV2VFQ&data=05%7C01%7Csmatysik%40nvidia.com%7C347ebe243c764d22761908dad7cbbad2%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638059567269062080%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=WmjhZpi1SocpDVAP5QtcM4kOQ6aiW%2FspDvMYPGwzXbQ%3D&reserved=0)

## Roadmap
* pre-alpha (Q4’22)
  - Goal: verify and adjust design, basic features, full-stability and performance are a non-goal
  - Contents: 
    - Plugin framework with extension modules discovery and loading (NF-001)
    - Codec registry with decoder\encoder matching mechanism (F-006)
    - Define plugin Interface(s) and implement at least two codecs plugins (Decoder, Encoder, Parser)
      - nvJpeg
      - nvJpeg2k
    - Unified API (NF-002)
    - Python bindings
* alpha (Q1’23)
  - Goal: integration-ready (DALI), all DALI must-haves implemented
  - Contents: 
    - Batch processing (F-002, F-003)
    - Partial decoding (F-008)
    - Decoupled API
    - Error and debug messages handling (F-004)
    - Custom allocators (F-009)
    - Plugins
      - nvTiff
      - FasterPng
    - Prepare to add custom plugins (NF-004)
    - Work with file-like API  (F-007)
* beta (Q2’23)
  - Goal: public release
  - Contents
    - Work with raw API (seek&load parts of data) (F-015)
    - Metadata parsing (F-005, F-013)
    - Performance optimization (PERF-001)
    - Adding wrapping plugins for 3rd party codecs

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
- CMake (3.14 or later)
- gcc 11.10
- NVIDIA CUDA 11.8 or greater
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
## Install

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
│       nvimgcdcs_module.h
│       nvimgcdcs_version.h
│       nvimgcodecs.h
│       nvjpeg.h
│       nvjpeg2k.h
│       nvjpeg2k_version.h
│
├───lib64
│       nvbmp.lib
│       nvimgcodecs.cp39-win_amd64.pyd
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
cmake --install . --config Release --prefix ..\bin
```

## Packaging

From a succesfully built project, installers can be generated using cpack:
```
cd build
cpack --config CPackConfig.cmake -DCMAKE_BUILD_TYPE=Release
```
This will generate in build directory *.zip or *tar.xz files

