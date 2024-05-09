..
  # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

.. _v0.3.0:

v0.3.0-beta.2
=============

This software is in beta version, which means it is still undergoing testing and development before its official release. It may contain bugs, errors, or incomplete features that could affect its performance and functionality. By using this software, you agree to accept the risks and limitations associated with beta software. We appreciate your feedback and suggestions to help us improve this software, but we do not guarantee that we will implement them or that the software will meet your expectations. Please use this software at your own discretion and responsibility.

Key Features and Enhancements
-----------------------------

* Added support for ROI in Python API  
* Enable runtime dynamic loading of nvJpeg2000 library
* Added support for nvJpeg2000 v0.8.0 
* Added GDPR required links to documentation

Fixed Issues
------------

* Fix validation of the whl fails https://github.com/NVIDIA/nvImageCodec/issues/2
* Fix incorrect decoding of 16-bit TIFF images https://github.com/NVIDIA/nvImageCodec/issues/5
* Fix encoding single channel images https://github.com/NVIDIA/nvImageCodec/issues/6
* Fix for passing cuda stream to `__dlpack__` function as a keyword only argument
* Fix missing synchronization with user cuda stream before decoding
* Fix shape returns wrong value for host Images
* Patch libtiff for CVE-2023-52356
* Patch libtiff for CVE-2023-6277
* Patch libtiff for CVE-2023-6228

