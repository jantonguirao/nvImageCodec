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

.. _v0.1.0-alpha.1:

v0.1.0-alpha.1
==============

NvImageCodecs-0.1.0-alpha.1 is the first release of nvImageCodec. This release is for evaluation purposes only.

Release Highlights
------------------

This nvImageCodec release includes the following key features:

* Batch processing
* Python bindings
* Sample applications
* API documentation
* Currently there are following codecs supported:
* nvjpeg
* nvjpeg2000

   * nvbmp (as an example extension module)
   * nvpnm (as an example extension module with encoder/writter only)
   * libturbo-jpeg (decoder only)
   * libtiff (decoder only)
   * opencv (decoders only)

Compatibility
-------------
This section highlights the compute stack nvImageCodec has been tested on

* Ubuntu x86 >= 20.04
* CUDA driver >= 11.8


Known Issues
------------
* Temporally 3rd party extensions are linked statically with particular codec libraries
* Temporally nvJpeg2000 extension is linked statically with nvJpeg2000 codec library 
 

License
-------
Nvidia Software Evaluation License
