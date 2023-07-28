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

.. _v0.1.0-alpha.2:

v0.1.0-alpha.2
==============

NvImageCodecs-0.1.0-alpha.2 is the second release of the project. This release is for evaluation purposes only.

Release Highlights
------------------

This nvImageCodecs release includes the following updates over v0.1.0-alpha.1:

* Fixed multiple logging when more than one instance of library is created
* Fixed synchronization issues with Image CUDA stream in GPU encoders
* Improved performance of single image pipeline
* Added parallel batch execution of decoding and encoding for bmp and pnm formats
* Added possibility to decode bmp to interleaved RGB
* Added possibility to pass NumPy array to decode function
* Added C and Python API reference documentation
* Added package with samples
* Improved testing
* C API improvements
    * Removed binary literals
    * Changed bool type to integer
    * Remove MCT struct and enable_color_transform parameters and use nvimgcdcsImageInfo_t instead
    * Added map and unmap to nvimgcdcsIOStreamDesc_t
* Python API improvements
    * Change decode and encode function which operates on files to read and write respectively
    * Replaced MCT and enable_color_conversion parameters with ColorSpec
    * Separate codec specific encode parameters in separate classes JpegEncodeParams and Jpeg2kEncodeParams      

Compatibility
-------------
This section highlights the compute stack nvImageCodecs has been tested on

* Ubuntu x86_x64 >= 20.04
* CUDA Toolkit >= 11.8
* GCC >= 9.4
* Python: 3.7, 3.8, 3.10

Refer to documentation of the sample applications for dependencies.

Known Issues and limitations
----------------------------
* Temporally 3rd party extensions are linked statically with particular codec libraries
* Temporally nvJpeg2000 extension is linked statically with nvJpeg2000 codec library 
* 3rd party extensions are excluded from EA packages 

License
-------
Nvidia Software Evaluation License
