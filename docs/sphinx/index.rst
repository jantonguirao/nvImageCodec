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

NVIDIA nvImageCodec
======================================
The nvImageCodec is a library of accelerated codec with unified interface. It is designed as a framework for extension modules which delivers codec plugins.

This nvImageCodec release includes the following key features:

* Unified API for decoding and encoding images
* Batch processing, with variable shape and heterogeneous formats images
* Codec prioritization with automatic fallback
* Builtin parsers for image format detection: jpeg, jpeg2000, tiff, bmp, png, pnm, webp 
* Python bindings
* Zero-copy interfaces to CV-CUDA, PyTorch and cuPy 
* End-end accelerated sample applications for common image transcoding

Currently there are following native codec extensions:

* nvjpeg_ext

   * Hardware jpeg decoder
   * CUDA jpeg decoder
   * CUDA lossless jpeg decoder
   * CUDA jpeg encoder

* nvjpeg2k_ext

   * CUDA jpeg 2000 decoder (including High Throughput jpeg2000)
   * CUDA jpeg 2000 encoder 

* nvbmp_ext

   * CPU bmp reader
   * CPU bmp writer

* nvpnm_ext

   * CPU pnm (ppm, pbm, pgm) writer

Additionally as a fallback there are following 3rd party codec extensions:

* libturbo-jpeg_ext

   * CPU jpeg decoder

* libtiff_ext 

   * CPU tiff decoder

* opencv_ext

   * CPU jpeg decoder
   * CPU jpeg2k_decoder
   * CPU png decoder
   * CPU bmp decoder
   * CPU pnm decoder
   * CPU tiff decoder
   * CPU webp decoder

.. toctree::
   :caption: Beginner's Guide
   :maxdepth: 3
   :hidden:
   
   Installation <installation>
   Getting Started <getting_started> 

.. toctree::
   :caption: API Reference
   :maxdepth: 3
   :hidden:

   C API <c_api>  
   Python API <py_api> 

.. toctree::
   :caption: Release Notes
   :maxdepth: 1
   :hidden:

   0.2.0-beta.1 <release_notes_v0.2.0>
   0.1.0-alpha.2 <release_notes_v0.1.0-alpha.2>
   0.1.0-alpha.1 <release_notes_v0.1.0-alpha.1>  

.. toctree::
   :caption: References
   :maxdepth: 1
   :hidden:
   
   GitHub <https://github.com/NVIDIA/nvImageCodec>  
   Reporting vulnerabilities <security>
   Acknowledgements <acknowledgements>
   License <license>
