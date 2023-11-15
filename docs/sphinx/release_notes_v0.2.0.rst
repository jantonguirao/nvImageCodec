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

.. _v0.2.0:

v0.2.0
==============

nvImageCodec v0.2.0-beta.1 is the first public release of the library.

Key Features and Enhancements
-----------------------------

This nvImageCodec release includes the following key features and enhancements:

* Changed name of library from nvImageCodecs to nvImageCodec
* Added CUDA 12.3 support
* Added support for arm64-sbsa
* Improved decode performance (e.g. nvJpeg2000 tiled)
* Improved testing
* Improved documentation 
    * Added simple sample C API usage
    * Added Python sample Jupyter notebooks to documentation  
* C API improvements
    * Adjusted cvcuda types adapter
    * Improved error handling and reporting 
    * Custom Executor can now be defined per Decoder/Encoder
    * Added possibility to pre-allocate resources
    * Added support for nvjpeg extra_flags option
* Python API improvements
    * Added support for Python 3.12 and deprecated Python 3.7  
    * Added support for DL-pack
    * Added support of __array_interface__
    * Added cpu() and cuda() convert methods to Python Image to enable transfers between Host and Device memory
    * Added as_images function
    * Added allow_any_depth to decode parameters
    * Added possibility to specify number of CPU threads to use in decoder/encoder
    * Added precision attribute to Image
    * Added __enter__ and __exit__ to Decoder and Encoder so they can be easily used with python "with" statement 
    * Python decode function now can convert to interleaved RGB and 8 bits by default
