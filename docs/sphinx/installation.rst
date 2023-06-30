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

This section describes the recommended dependencies to use nvImageCodecs

* Ubuntu >= 20.04
* CUDA driver >= 11.8
* CUDA Toolit >= 11.8

Setup
-----

The following steps describe how to install nvImageCodecs from pre-built install packages. Choose the installation method that meets your environment needs.

Download the nvImageCodecs tar/deb package from `here <https://github.com/xxxTODOxxx/releases/tag/v0.1.0-alpha.1>`_

* Tar File Installation

    Navigate to directory containing the nvImageCodecs tar file.

    Unzip the nvImageCodecs runtime and developer package: ::

        tar -xvf nvimgcodecs-0.1.0_alpha.1-cuda12-x86_64-linux-lib.tar.gz -C /opt/nvidia/

    Optionally Unzip the tests. ::

        tar -xvf nvimgcodecs-0.1.0_alpha.1-cuda12-x86_64-linux-tests.tar.gz -C /opt/nvidia/

* Debian Local Installation

    Navigate to directory containing the nvImageCodecs tar file.

    Install the nvImageCodecs runtime and developer package: ::

        sudo apt-get install -y ./nvimgcodecs-0.1.0_alpha.1-cuda12-x86_64-linux-lib.deb

    Optionally install the tests. ::

        sudo apt-get install -y ./nvimgcodecs-0.1.0_alpha.1-cuda12-x86_64-linux-tests.deb

* Python WHL File Installation. ::

    pip install nvidia_nvimgcodecs_cuda120-0.1.0-py3-none-manylinux2014_x86_64.whl

* Running the samples. ::

    Follow the instructions written in the README.md file of the samples directory.
