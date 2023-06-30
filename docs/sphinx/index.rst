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

NVIDIA nvImageCodecs
======================================
The nvImageCodecs is a library of accelerated codecs with unified interface. It is designed as a framework for extension modules which delivers codec plugins.

This nvImageCodecs release includes the following key features:

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

Notice
--------------------
The information provided in this specification is believed to be accurate and reliable as of the date provided. However, NVIDIA Corporation (“NVIDIA”) does not give any representations or warranties, expressed or implied, as to the accuracy or completeness of such information. NVIDIA shall have no liability for the consequences or use of such information or for any infringement of patents or other rights of third parties that may result from its use. This publication supersedes and replaces all other specifications for the product that may have been previously supplied.

NVIDIA reserves the right to make corrections, modifications, enhancements, improvements, and other changes to this specification, at any time and/or to discontinue any product or service without notice. Customer should obtain the latest relevant specification before placing orders and should verify that such information is current and complete.

NVIDIA products are sold subject to the NVIDIA standard terms and conditions of sale supplied at the time of order acknowledgement, unless otherwise agreed in an individual sales agreement signed by authorized representatives of NVIDIA and customer. NVIDIA hereby expressly objects to applying any customer general terms and conditions with regards to the purchase of the NVIDIA product referenced in this specification.

NVIDIA products are not designed, authorized or warranted to be suitable for use in medical, military, aircraft, space or life support equipment, nor in applications where failure or malfunction of the NVIDIA product can reasonably be expected to result in personal injury, death or property or environmental damage. NVIDIA accepts no liability for inclusion and/or use of NVIDIA products in such equipment or applications and therefore such inclusion and/or use is at customer’s own risk.

NVIDIA makes no representation or warranty that products based on these specifications will be suitable for any specified use without further testing or modification. Testing of all parameters of each product is not necessarily performed by NVIDIA. It is customer’s sole responsibility to ensure the product is suitable and fit for the application planned by customer and to do the necessary testing for the application in order to avoid a default of the application or the product. Weaknesses in customer’s product designs may affect the quality and reliability of the NVIDIA product and may result in additional or different conditions and/or requirements beyond those contained in this specification. NVIDIA does not accept any liability related to any default, damage, costs or problem which may be based on or attributable to: (i) the use of the NVIDIA product in any manner that is contrary to this specification, or (ii) customer product designs.

No license, either expressed or implied, is granted under any NVIDIA patent right, copyright, or other NVIDIA intellectual property right under this specification. Information published by NVIDIA regarding third-party products or services does not constitute a license from NVIDIA to use such products or services or a warranty or endorsement thereof. Use of such information may require a license from a third party under the patents or other intellectual property rights of the third party, or a license from NVIDIA under the patents or other intellectual property rights of NVIDIA. Reproduction of information in this specification is permissible only if reproduction is approved by NVIDIA in writing, is reproduced without alteration, and is accompanied by all associated conditions, limitations, and notices.

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE. Notwithstanding any damages that customer might incur for any reason whatsoever, NVIDIA’s aggregate and cumulative liability towards customer for the products described herein shall be limited in accordance with the NVIDIA terms and conditions of sale for the product.


Trademarks
--------------------

NVIDIA, the NVIDIA logo, NVIDIA CV-CUDA, and NVIDIA TensorRT are trademarks and/or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.


Copyright
--------------------
© 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.    

.. toctree::
   :caption: Beginner's Guide
   :maxdepth: 1
   :hidden:
   
   Installation <installation>
   Getting Started <getting_started> 

.. toctree::
   :caption: API Reference
   :maxdepth: 2
   :hidden:

   C API <c_api>  
   Python API <py_api> 

.. toctree::
   :caption: Release Notes
   :maxdepth: 1
   :hidden:

   0.1.0-alpha.1 <release_notes_v0.1.0-alpha.1>  

.. toctree::
   :caption: References
   :maxdepth: 1
   :hidden:

   License <license>
