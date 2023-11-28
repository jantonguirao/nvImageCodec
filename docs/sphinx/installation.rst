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
    * Ubuntu x86_64 >= 20.04
    * WSL2 with Ubuntu >= 20.04
* NVIDIA driver >= 520.56.06
* CUDA Toolit >= 11.8
* GCC >= 9.4
* Python >= 3.8
* cmake >= 3.18

Setup
-----

The following steps describe how to install nvImageCodec from pre-built install packages. Choose the installation method that meets your environment needs. The `x` letter in the below command is the build id. It will be 0 when the package is built locally.

Download the nvImageCodec tar/deb package from `here <https://github.com/NVIDIA/nvImageCodec/releases>`_

* Tar File Installation

    Navigate to directory containing the nvImageCodec tar file.

    Unzip the nvImageCodec runtime and developer package: ::

        tar -xvf nvimgcodec-0.2.0.x-cuda12-x86_64-linux-lib.tar.gz -C /opt/nvidia/

* Debian Local Installation

    Navigate to directory containing the nvImageCodec tar file.

    Install the nvImageCodec runtime and developer package: ::

        sudo apt-get install -y ./nvimgcodec-0.2.0.x-cuda12-x86_64-linux-lib.deb

* Python WHL File Installation. ::

    pip install nvidia_nvimgcodec_cu12-0.2.0.x-py3-none-manylinux2014_x86_64.whl

* Running the samples. ::

    Follow the instructions written in the README.md file of the samples directory.
