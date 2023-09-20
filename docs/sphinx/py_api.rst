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
  # limitations under the License

.. _py_api:

Python API
==========
 
.. This is the Python API reference for the NVIDIA® nvImageCodecs library.

.. automodule:: nvidia.nvimgcodecs
   :members:

Backend
-------

.. autoclass:: Backend
   :members:
   :special-members: __init__

BackendKind
-----------

.. autoclass:: BackendKind
   :members:
   :exclude-members: name

BackendParams
-------------

.. autoclass:: BackendParams
   :members:
   :special-members: __init__


ChromaSubsampling
-----------------

.. autoclass:: ChromaSubsampling
   :members:
   :exclude-members: name

DecodeParams
------------

.. autoclass:: DecodeParams
   :members:
   :special-members: __init__

Decoder
-------

.. autoclass:: Decoder
   :members:
   :special-members: __init__


JpegEncodeParams
----------------

.. autoclass:: JpegEncodeParams
   :members:
   :special-members: __init__

Jpeg2kEncodeParams
------------------

.. autoclass:: Jpeg2kEncodeParams
   :members:
   :special-members: __init__

EncodeParams
------------

.. autoclass:: EncodeParams
   :members:
   :special-members: __init__

Encoder
-------

.. autoclass:: Encoder
   :members:
   :special-members: __init__  

ImageBufferKind
---------------

.. autoclass:: ImageBufferKind
   :members:
   :exclude-members: name

Image
-----

.. autoclass:: Image
   :members:
   :undoc-members:
   :special-members: __cuda_array_interface__, __dlpack__, __dlpack_device__

Jpeg2kBitstreamType
-------------------

.. autoclass:: Jpeg2kBitstreamType
   :members:
   :exclude-members: name

Jpeg2kProgOrder
---------------

.. autoclass:: Jpeg2kProgOrder
   :members:
   :exclude-members: name

ColorSpec
---------

.. autoclass:: ColorSpec
   :members:
   :exclude-members: name

as_image
---------

.. autofunction:: as_image

from_dlpack
-----------

.. autofunction:: from_dlpack

.. toctree::
   :caption: Python API
   :maxdepth: 2
   :hidden:
