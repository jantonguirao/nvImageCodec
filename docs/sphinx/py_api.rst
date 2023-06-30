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
 
This is the Python API reference for the NVIDIA® nvImageCodecs library.

Backend
-------

**class nvidia.nvimgcodecs.Backend**

   ``property backend_kind``

      Backend kind (e.g. GPU_ONLY or CPU_ONLY).

   ``property backend_params``

      Backend parameters.

   ``property load_hint``

      Fraction of the batch samples that will be picked by this backend. The remaining samples will be picked by the next lower priority backend.

BackendKind
-----------

**class nvidia.nvimgcodecs.BackendKind**

   Members:

   ``CPU_ONLY``

   ``GPU_ONLY``

   ``HYBRID_CPU_GPU``

   ``HW_GPU_ONLY``

BackendParams
-------------

**class nvidia.nvimgcodecs.BackendParams**

   ``property load_hint``

      Fraction of the batch samples that will be picked by this backend. The remaining samples will be picked by the next lower priority backend. This is just hint so particular codec can ignore this value

ChromaSubsampling
-----------------

**class nvidia.nvimgcodecs.ChromaSubsampling**

   Members:

   ``CSS_444``

   ``CSS_422``

   ``CSS_420``

   ``CSS_440``

   ``CSS_411``

   ``CSS_410``

   ``CSS_GRAY``

   ``CSS_410V``

DecodeParams
------------

**class nvidia.nvimgcodecs.DecodeParams**

   ``property enable_color_conversion``

      Enable color conversion to RGB

   ``property enable_orientation``

      Apply EXIF orientation if available

Decoder
-------

**class nvidia.nvimgcodecs.Decoder**

   **decode(*args, **kwargs)**

      Overloaded function.

      1. **decode(self: nvidia.nvimgcodecs.Decoder, data: bytes, params: nvidia.nvimgcodecs.DecodeParams = nvidia.nvimgcodecs.DecodeParams(), cuda_stream: int = 0) -> nvidia.nvimgcodecs.Image**

             Executes decoding of data.

             Args:
                ``data``: Buffer with bytes to decode.

                ``params``: Decode parameters.

                ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

             Returns:
                ``nvidia.nvimgcodecs.Image``: Decoded image.

      2. **decode(self: nvidia.nvimgcodecs.Decoder, file_name: str, params: nvidia.nvimgcodecs.DecodeParams = nvidia.nvimgcodecs.DecodeParams(), cuda_stream: int = 0) -> nvidia.nvimgcodecs.Image**


             Executes decoding of file.

             Args:
                ``file_name``: File name to decode.

                ``params``: Decode parameters.

                ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

             Returns:
                ``nvidia.nvimgcodecs.Image``: Decoded image.

      3. **decode(self: nvidia.nvimgcodecs.Decoder, file_names: List[bytes], params: nvidia.nvimgcodecs.DecodeParams = nvidia.nvimgcodecs.DecodeParams(), cuda_stream: int = 0) -> List[nvidia.nvimgcodecs.Image]**


             Executes data batch decoding.

             Args:
                ``file_names``: List of buffers with code streams to decode.

                ``params``: Decode parameters.

                ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

             Returns:
                ``List[nvidia.nvimgcodecs.Image]``:  List of decoded images.

      4. **decode(self: nvidia.nvimgcodecs.Decoder, data_list: List[str], params: nvidia.nvimgcodecs.DecodeParams = nvidia.nvimgcodecs.DecodeParams(), cuda_stream: int = 0) -> List[nvidia.nvimgcodecs.Image]**


             Executes file batch decoding.

             Args:
                ``data_list``: List of file names to decode.

                ``params``: Decode parameters.

                ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

             Returns:
                ``List[nvidia.nvimgcodecs.Image]``: List of decoded images.

EncodeParams
------------

**class nvidia.nvimgcodecs.EncodeParams**

   ``property chroma_subsampling``

      Chroma subsampling (default ChromaSubsampling.CSS_444)

   ``property jpeg2k_bitstream_type``

      Jpeg 2000 bitstream type (default JP2)

   ``property jpeg2k_code_block_size``

      Jpeg 2000 code block width and height (default 64x64)

   ``property jpeg2k_num_resolutions``

      Jpeg 2000 number of resolutions - decomposition levels (default 5)

   ``property jpeg2k_prog_order``

      Jpeg 2000 progression order (default RPCL)

   ``property jpeg2k_reversible``

      Use reversible Jpeg 2000 transform (default False)

   ``property jpeg_optimized_huffman``

      Use Jpeg encoding with optimized Huffman (default False)

   ``property jpeg_progressive``

      Use Jpeg progressive encoding (default False)

   ``property mct_mode``

      Multi-Color Transform mode value (default MctMode.RGB)

   ``property quality``

      Quality value 0-100 (default 95)

   ``property target_psnr``

      Target psnr (default 50)

Encoder
-------

**class nvidia.nvimgcodecs.Encoder**

   **encode(*args, **kwargs)**

      Overloaded function.

      1. **encode(self: nvidia.nvimgcodecs.Encoder, image: nvidia.nvimgcodecs.Image, codec: str, params: nvidia.nvimgcodecs.EncodeParams = nvidia.nvimgcodecs.EncodeParams(), cuda_stream: int = 0) -> bytes**


             Encode image to buffer.

             Args:
                ``image``: Image to encode

                ``codec``: String that defines the output format e.g.’jpeg2k’. When it is file extension it must include a leading period e.g. ‘.jp2’.

                ``params``: Encode parameters.

                ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

             Returns:
                ``bytes``: Buffer with compressed code stream.

      2. **encode(self: nvidia.nvimgcodecs.Encoder, file_name: str, image: nvidia.nvimgcodecs.Image, codec: str = ‘’, params: nvidia.nvimgcodecs.EncodeParams = nvidia.nvimgcodecs.EncodeParams(), cuda_stream: int = 0) -> None**


             Encode image to file.

             Args:
                ``file_name``: File name to save encoded code stream. 

                ``image``: Image to encode
                
                ``codec`` (optional): String that defines the output format e.g.’jpeg2k’. When it is file extension it must include a leading period e.g. ‘.jp2’. If codec is not specified, it is deducted based on file extension. If there is no extension by default ‘jpeg’ is chosen.

                ``params``: Encode parameters.

                ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

             Returns:
                ``None``

      3. **encode(self: nvidia.nvimgcodecs.Encoder, images: List[nvidia.nvimgcodecs.Image], codec: str, params: nvidia.nvimgcodecs.EncodeParams = nvidia.nvimgcodecs.EncodeParams(), cuda_stream: int = 0) -> List[bytes]**


             Encode batch of images to buffers.

             Args:
                ``images``: List of images to encode

                ``codec``: String that defines the output format e.g.’jpeg2k’. When it is file extension it must include a leading period e.g. ‘.jp2’.
                
                ``params``: Encode parameters.
                
                ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

             Returns:
                ``List[bytes]``: List of buffers with compressed code streams.

      4. **encode(self: nvidia.nvimgcodecs.Encoder, file_names: List[str], images: List[nvidia.nvimgcodecs.Image], codec: str = ‘’, params: nvidia.nvimgcodecs.EncodeParams = nvidia.nvimgcodecs.EncodeParams(), cuda_stream: int = 0) -> None**


             Encode batch of images to files.

             Args:
                ``images``: List of images to encode
                
                ``file_names``: List of file names to save encoded code streams.
                
                ``codec`` (optional): String that defines the output format e.g.’jpeg2k’. When it is file extension it must include a leading period e.g. ‘.jp2’. If codec is not specified, it is deducted based on file extension. If there is no extension by default ‘jpeg’ is chosen.

                ``params``: Encode parameters.

                ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

             Returns:
                ``List[bytes]``: List of buffers with compressed code streams.

Image
-----

**class nvidia.nvimgcodecs.Image**

   ``property dtype``

   ``property height``

   ``property ndim``

   ``property shape``

   ``property width``

Jpeg2kBitstreamType
-------------------

**class nvidia.nvimgcodecs.Jpeg2kBitstreamType**

   Members:

   ``J2K``

   ``JP2``

Jpeg2kProgOrder
---------------

**class nvidia.nvimgcodecs.Jpeg2kProgOrder**

   Jpeg2000 Progression order 

   Members:

   ``LRCP``

   ``RLCP``

   ``RPCL``

   ``PCRL``

   ``CPRL``

MctMode
-------

**class nvidia.nvimgcodecs.MctMode**

   Members:

   ``YCC``

   ``RGB``

as_image
--------

**nvidia.nvimgcodecs.as_image(source_buffer: handle, cuda_stream: int = 0) -> nvimgcdcs::Image**

Wrap an external buffer as an image and tie the buffer lifetime to the image

Args:
    ``source_buffer``: Object with __cuda_array_interface__.

    ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place in created Image.

Returns:
    ``nvidia.nvimgcodecs.Image``: Image which wraps provided source_buffer.

.. toctree::
    :caption: Python API
    :maxdepth: 2
    :hidden:
