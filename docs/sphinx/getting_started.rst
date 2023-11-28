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

.. _getting_started:

Getting Started
===============

This section guides you step-by-step how to decode and encode images on the GPU using the nvImageCodec APIs. Before getting started, please review the :ref:`pre-requisites <prerequisites>`. Once reviewed, head over to the :ref:`samples <samples>`' section which showcases various nvImageCodec samples.

.. note:: Throughout this document, the terms "CPU" and "Host" are used synonymously. Similarly, the terms "GPU" and "Device" are synonymous.

Thread Safety
-------------

Not all nvImageCodec types are thread safe. For user-provided allocators (fields in ``nvimgcodecExecutionParams_t`` structure), the user needs to ensure thread safety. 

.. _prerequisites:

Pre-requisites
--------------

Following are the required dependencies to compile nvImageCodec samples.

* Ubuntu >= 20.04
* NVIDIA driver >= 520.56.06
* CUDA Toolit >= 11.8
* Supported systems:
    * Linux (Ubuntu >= 20.04, RHEL >= 7, and other PEP599 manylinux 2014 compatible platforms)
    * WSL2 with Ubuntu >= 20.04

C++ Samples' Dependencies:

* CMake >= 3.18
* gcc >= 9.4

Python Samples' Dependencies:

* Torch
* Torchvision
* cuPy
* cuCIM
* CV-CUDA

.. _samples:

Samples
-------

The next section documents the samples showing various use cases across two different types APIs available to consume nvImageCodec functionality:

.. toctree::
   :maxdepth: 4

    C API samples <samples/c_samples>
    Python API samples <samples/python_samples>

Refer to the :ref:`Installation` docs for the sample installation guide using ``*.deb`` or ``*.tar`` installers.
Refer to the sample README for instructions to compile samples from the source.


