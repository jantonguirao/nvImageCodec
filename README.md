# nvImageCodecs

## Description

nvImageCodecs is a library of accelerated codecs with unified interface. It is designed as a framework for extension modules which delivers particular codec plugins.

## Prototype
Currently this project is in prototype state  with initial support for:
- nvJPEG
- nvJPEG2000
- bmp as a example extension module

There are following known limitation:
- Only planar RGB format is supported as a input for encoder and output from decoder
- There is limited set of supported decoder and encoder parameters
- No custom allocators yet
- No batch processing yet
- No ROI support yet
- No Metadata support yet (appart of EXIF orientation for jpeg)

## Requirements
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
