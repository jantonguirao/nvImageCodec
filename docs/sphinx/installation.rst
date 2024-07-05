..
  # SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  # http://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.

.. _installation:

Installation
============

Pre-requisites
--------------

This section describes the recommended dependencies to use nvImageCodec.

* Linux distro:
   * x86_64
      * Debian 11, 12
      * Fedora 39
      * RHEL 8, 9
      * OpenSUSE 15
      * SLES 15
      * Ubuntu 20.04, 22.04
      * WSL2 Ubuntu 20.04
   * arm64-sbsa
      * RHEL 8, 9
      * SLES 15
      * Ubuntu 20.04, 22.04
   * aarch64-jetson (CUDA Toolkit >= 12.0)
      * Ubuntu 22.04
* Windows
   * x86_64
      * `Microsoft Visual C++ Redistributable <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>`_  
* NVIDIA driver >= 520.56.06
* CUDA Toolkit > = 11.8
* Python >= 3.8
* GCC >= 9.4
* cmake >= 3.18

Install nvImageCodec library
----------------------------

You can download and install the appropriate built binary packages from the `nvImageCodec Developer Page <https://developer.nvidia.com/nvimgcodec-downloads>`_ or install nvImageCodec Python from PyPI as it is described below.

* nvImageCodec Python for CUDA 11.x ::

    pip install nvidia-nvimgcodec-cu11

* nvImageCodec Python for CUDA 12.x ::

    pip install nvidia-nvimgcodec-cu12

Optional installation of nvJPEG library
---------------------------------------

If you do not have CUDA Toolkit installed, or you would like install nvJPEG library independently, you can use the instructions described below.

* Install the nvidia-pyindex module ::

    pip install nvidia-pyindex

* Install nvJPEG for CUDA 11.x ::

    pip install nvidia-nvjpeg-cu11

* Install nvJPEG for CUDA 12.x ::

    pip install nvidia-nvjpeg-cu12

Optional installation of nvJPEG2000 library
-------------------------------------------
`nvJPEG2000 library <https://developer.nvidia.com/nvjpeg2000-downloads>`_ can be installed in the system, or installed as a Python package. For the latter, follow the instructions below.

* Install the nvidia-pyindex module ::

    pip install nvidia-pyindex

* Install nvJPEG2000 for CUDA 11.x ::

    pip install nvidia-nvjpeg2k-cu11

* Install nvJPEG2000 for CUDA 12.x ::

    pip install nvidia-nvjpeg2k-cu12

* Install nvJPEG2000 for CUDA 12.x on Tegra platforms ::

    pip install nvidia-nvjpeg2k-tegra-cu12

Please see also `nvJPEG2000 installation documentation <https://docs.nvidia.com/cuda/nvjpeg2000/userguide.html#installing-nvjpeg2000>`_ for more information
