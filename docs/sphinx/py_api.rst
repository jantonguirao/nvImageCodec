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

    .. py:class:: nvidia.nvimgcodecs.Backend

        Backend for decoding or encoding.

        .. py:method:: __init__(*args, **kwargs)

            Overloaded function.

            1. __init__(self: nvidia.nvimgcodecs.Backend) -> None
                
                Default constructor

            2. __init__(self: nvidia.nvimgcodecs.Backend, backend_kind: nvidia.nvimgcodecs.BackendKind, load_hint: float = 1.0) -> None
                
                Constructor with backend kind and loadhint parameters.

            3. __init__(self: nvidia.nvimgcodecs.Backend, backend_kind: nvidia.nvimgcodecs.BackendKind, backend_params: nvidia.nvimgcodecs.BackendParams) -> None
                
                Constructor with backend kind and backend parameters       

        .. py:attribute:: backend_kind

            Backend kind (e.g. GPU_ONLY or CPU_ONLY).

        .. py:attribute:: backend_params

            Backend parameters.

        .. py:attribute:: load_hint

            Fraction of the batch samples that will be picked by this backend. The remaining samples will be picked by the next lower priority backend.

BackendKind
-----------

    .. py:class:: nvidia.nvimgcodecs.BackendKind

        Backend kind for decoding or encoding.

        .. py:attribute:: CPU_ONLY

        .. py:attribute:: GPU_ONLY

        .. py:attribute:: HYBRID_CPU_GPU

        .. py:attribute:: HW_GPU_ONLY

BackendParams
-------------

    .. py:class:: nvidia.nvimgcodecs.BackendParams

        Backend parameters.

        .. py:method:: __init__(*args, **kwargs)

            Overloaded function.

            1. __init__(self: nvidia.nvimgcodecs.BackendParams) -> None
                
                Default constructor

            2. __init__(self: nvidia.nvimgcodecs.BackendParams, load_hint: bool = 1.0) -> None
                
                Constructor with load hint parameters 

        .. py:attribute:: load_hint

            Fraction of the batch samples that will be picked by this backend. The remaining samples will be picked by the next lower priority backend. This is just hint so particular codec can ignore this value

ChromaSubsampling
-----------------

    .. py:class:: nvidia.nvimgcodecs.ChromaSubsampling

        Chroma subsampling.

        .. py:attribute:: CSS_444

        .. py:attribute:: CSS_422

        .. py:attribute:: CSS_420

        .. py:attribute:: CSS_440

        .. py:attribute:: CSS_411

        .. py:attribute:: CSS_410

        .. py:attribute:: CSS_GRAY

        .. py:attribute:: CSS_410V

DecodeParams
------------

    .. py:class:: nvidia.nvimgcodecs.DecodeParams

        Decode parameters.

        .. py:method:: __init__(*args, **kwargs)

            Overloaded function.

            1. __init__(self: nvidia.nvimgcodecs.DecodeParams) -> None
                
                Default constructor

            2. __init__(self: nvidia.nvimgcodecs.DecodeParams, apply_exif_orientation: bool = True, enable_color_conversion: bool = True) -> None
                
                Constructor with apply_exif_orientation and enable_color_conversion parameters 

        .. py:attribute:: enable_color_conversion

            Enable color conversion to RGB

        .. py:attribute:: apply_exif_orientation

            Apply EXIF orientation if available

Decoder
-------

    .. py:class:: nvidia.nvimgcodecs.Decoder

        Generic image decoder.

        .. py:method:: __init__(*args, **kwargs)

            Overloaded function.

                1. __init__(self: nvidia.nvimgcodecs.Decoder, device_id: int = -1, backends: List[nvidia.nvimgcodecs.Backend] = [], options: str = ‘:fancy_upsampling=0’) -> None
                        
                        Initialize decoder.

                        **Args:**

                            ``device_id``: Device id to execute decoding on. As default, it is current device. (device_id = -1)

                            ``backends``: List of allowed backends. If empty, all backends are allowed with default parameters.

                            ``options``: Decoder specific options e.g.: “nvjpeg:fancy_upsampling=1”

                2. __init__(self: nvidia.nvimgcodecs.Decoder, device_id: int = -1, backend_kinds: List[nvidia.nvimgcodecs.BackendKind] = [], options: str = ‘:fancy_upsampling=0’) -> None
                        
                        Initialize decoder.

                        **Args:**

                            ``device_id``: Device id to execute decoding on.  As default, it is current device. (device_id = -1)

                            ``backend_kinds``: List of allowed backend kinds. If empty or None, all backends are allowed with default parameters.

                            ``options``: Decoder specific options e.g.: “nvjpeg:fancy_upsampling=1”  

        .. py:method:: decode(*args, **kwargs)

            Overloaded function.

            1. **decode(self: nvidia.nvimgcodecs.Decoder, data: bytes, params: nvidia.nvimgcodecs.DecodeParams = nvidia.nvimgcodecs.DecodeParams(), cuda_stream: int = 0) -> nvidia.nvimgcodecs.Image**
                    
                    Executes decoding of data.

                    **Args:**

                        ``data``: Buffer with bytes to decode.

                        ``params``: Decode parameters.

                        ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

                    **Returns:**

                        ``nvidia.nvimgcodecs.Image``: Decoded image.

            3. **decode(self: nvidia.nvimgcodecs.Decoder, file_names: List[bytes], params: nvidia.nvimgcodecs.DecodeParams = nvidia.nvimgcodecs.DecodeParams(), cuda_stream: int = 0) -> List[nvidia.nvimgcodecs.Image]**
                    
                    Executes data batch decoding.

                    **Args:**

                        ``file_names``: List of buffers with code streams to decode.

                        ``params``: Decode parameters.

                        ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

                    **Returns:**

                        ``List[nvidia.nvimgcodecs.Image]``:  List of decoded images.

        .. py:method:: read(*args, **kwargs)

            Overloaded function.

            1. **read(self: nvidia.nvimgcodecs.Decoder, file_name: str, params: nvidia.nvimgcodecs.DecodeParams = nvidia.nvimgcodecs.DecodeParams(), cuda_stream: int = 0) -> nvidia.nvimgcodecs.Image**
                    
                    Read file and decode it.

                    **Args:**

                        ``file_name``: File name to decode.

                        ``params``: Decode parameters.

                        ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

                    **Returns:**

                        ``nvidia.nvimgcodecs.Image``: Decoded image.

            2. **read(self: nvidia.nvimgcodecs.Decoder, data_list: List[str], params: nvidia.nvimgcodecs.DecodeParams = nvidia.nvimgcodecs.DecodeParams(), cuda_stream: int = 0) -> List[nvidia.nvimgcodecs.Image]**
                    
                    Read batch of files and decode them.

                    **Args:**

                        ``data_list``: List of file names to decode.

                        ``params``: Decode parameters.

                        ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

                    **Returns:**

                        ``List[nvidia.nvimgcodecs.Image]``: List of decoded images.

EncodeParams
------------

    .. py:class:: nvidia.nvimgcodecs.EncodeParams

        Encode parameters.

        .. py:method:: __init__(*args, **kwargs)

            Overloaded function.

               1. __init__(self: nvidia.nvimgcodecs.EncodeParams) -> None
               
                   Default constructor

               2. __init__(self: nvidia.nvimgcodecs.EncodeParams, quality: float = 95, target_psnr: float = 50, mct_mode: nvidia.nvimgcodecs.MctMode = <MctMode.RGB: 1>, chroma_subsampling: nvidia.nvimgcodecs.ChromaSubsampling = <ChromaSubsampling.CSS_444: 0>, jpeg_progressive: bool = False, jpeg_optimized_huffman: bool = False, jpeg2k_reversible: bool = False, jpeg2k_code_block_size: Tuple[int, int] = (64, 64), jpeg2k_num_resolutions: int = 5, jpeg2k_bitstream_type: nvidia.nvimgcodecs.Jpeg2kBitstreamType = <Jpeg2kBitstreamType.JP2: 1>, jpeg2k_prog_order: nvidia.nvimgcodecs.Jpeg2kProgOrder = <Jpeg2kProgOrder.RPCL: 2>) -> None
               
                   Constructor with quality, target_psnr, mct_mode, chroma_subsampling etc. parameters 

        .. py:attribute:: chroma_subsampling

            Chroma subsampling (default ChromaSubsampling.CSS_444)

        .. py:attribute:: jpeg2k_bitstream_type

            Jpeg 2000 bitstream type (default JP2)

        .. py:attribute:: jpeg2k_code_block_size

            Jpeg 2000 code block width and height (default 64x64)

        .. py:attribute:: jpeg2k_num_resolutions

            Jpeg 2000 number of resolutions - decomposition levels (default 5)

        .. py:attribute:: jpeg2k_prog_order

            Jpeg 2000 progression order (default RPCL)

        .. py:attribute:: jpeg2k_reversible

            Use reversible Jpeg 2000 transform (default False)

        .. py:attribute:: jpeg_optimized_huffman

            Use Jpeg encoding with optimized Huffman (default False)

        .. py:attribute:: jpeg_progressive

            Use Jpeg progressive encoding (default False)

        .. py:attribute:: mct_mode

            Multi-Color Transform mode value (default MctMode.RGB)

        .. py:attribute:: quality

            Quality value 0-100 (default 95)

        .. py:attribute:: target_psnr

            Target psnr (default 50)

Encoder
-------

    .. py:class:: nvidia.nvimgcodecs.Encoder

        Generic image encoder.

        .. py:method:: __init__(*args, **kwargs)

            Overloaded function.

            1. __init__(self: nvidia.nvimgcodecs.Encoder, device_id: int = -1, backends: List[nvidia.nvimgcodecs.Backend] = [], options: str = ‘’) -> None
                
                Initialize encoder.

                    **Args:**

                        ``device_id``: Device id to execute encoding on. As default, it is current device. (device_id = -1)

                        ``backends``: List of allowed backends. If empty, all backends are allowed with default parameters.

                        ``options``: Encoder specific options.

            2. __init__(self: nvidia.nvimgcodecs.Encoder, device_id: int = -1, backend_kinds: List[nvidia.nvimgcodecs.BackendKind] = [], options: str = ‘:fancy_upsampling=0’) -> None
                
                Initialize encoder.

                    **Args:**

                        ``device_id``: Device id to execute encoding on. As default, it is current device. (device_id = -1)

                        ``backend_kinds``: List of allowed backend kinds. If empty or None, all backends are allowed with default parameters.
                        
                        ``options``: Encoder specific options.       

        .. py:method:: encode(*args, **kwargs)

            Overloaded function.

            1. **encode(self: nvidia.nvimgcodecs.Encoder, image: nvidia.nvimgcodecs.Image, codec: str, params: nvidia.nvimgcodecs.EncodeParams = nvidia.nvimgcodecs.EncodeParams(), cuda_stream: int = 0) -> bytes**
                    
                    Encode image to buffer.

                    **Args:**

                        ``image``: Image to encode

                        ``codec``: String that defines the output format e.g.’jpeg2k’. When it is file extension it must include a leading period e.g. ‘.jp2’.

                        ``params``: Encode parameters.

                        ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

                    **Returns:**

                        ``bytes``: Buffer with compressed code stream.


            2. **encode(self: nvidia.nvimgcodecs.Encoder, images: List[nvidia.nvimgcodecs.Image], codec: str, params: nvidia.nvimgcodecs.EncodeParams = nvidia.nvimgcodecs.EncodeParams(), cuda_stream: int = 0) -> List[bytes]**
                    
                    Encode batch of images to buffers.

                    **Args:**

                        ``images``: List of images to encode

                        ``codec``: String that defines the output format e.g.’jpeg2k’. When it is file extension it must include a leading period e.g. ‘.jp2’.
                        
                        ``params``: Encode parameters.
                        
                        ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

                    **Returns:**

                        ``List[bytes]``: List of buffers with compressed code streams.

        .. py:method:: write(*args, **kwargs)

            Overloaded function.

            1. **write(self: nvidia.nvimgcodecs.Encoder, file_name: str, image: nvidia.nvimgcodecs.Image, codec: str = ‘’, params: nvidia.nvimgcodecs.EncodeParams = nvidia.nvimgcodecs.EncodeParams(), cuda_stream: int = 0) -> None**
                    
                    Encode image and write to file.

                    **Args:**

                        ``file_name``: File name to save encoded code stream. 

                        ``image``: Image to encode
                        
                        ``codec`` (optional): String that defines the output format e.g.’jpeg2k’. When it is file extension it must include a leading period e.g. ‘.jp2’. If codec is not specified, it is deducted based on file extension. If there is no extension by default ‘jpeg’ is chosen.

                        ``params``: Encode parameters.

                        ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

                    **Returns:**

                        ``None``

            2. **write(self: nvidia.nvimgcodecs.Encoder, file_names: List[str], images: List[nvidia.nvimgcodecs.Image], codec: str = ‘’, params: nvidia.nvimgcodecs.EncodeParams = nvidia.nvimgcodecs.EncodeParams(), cuda_stream: int = 0) -> None**
                    
                    Encode batch of images and write to files.

                    **Args:**

                        ``images``: List of images to encode
                        
                        ``file_names``: List of file names to save encoded code streams.
                        
                        ``codec`` (optional): String that defines the output format e.g.’jpeg2k’. When it is file extension it must include a leading period e.g. ‘.jp2’. If codec is not specified, it is deducted based on file extension. If there is no extension by default ‘jpeg’ is chosen.

                        ``params``: Encode parameters.

                        ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place.

                    **Returns:**

                        ``List[bytes]``: List of buffers with compressed code streams.

Image
-----

    .. py:class:: nvidia.nvimgcodecs.Image

        Class which wraps buffer with pixels. It can be decoded pixels or pixels to encode.

        .. py:attribute:: dtype

        .. py:attribute:: height

        .. py:attribute:: ndim

        .. py:attribute:: shape

        .. py:attribute:: width

Jpeg2kBitstreamType
-------------------

    .. py:class:: nvidia.nvimgcodecs.Jpeg2kBitstreamType

        Jpeg2000 bitstream type

        .. py:attribute:: J2K

        .. py:attribute:: JP2

Jpeg2kProgOrder
---------------

    .. py:class:: nvidia.nvimgcodecs.Jpeg2kProgOrder

        Jpeg2000 Progression order 

        .. py:attribute:: LRCP

        .. py:attribute:: RLCP

        .. py:attribute:: RPCL

        .. py:attribute:: PCRL

        .. py:attribute:: CPRL

MctMode
-------

    .. py:class:: nvidia.nvimgcodecs.MctMode

        Multi-Color transform mode

        .. py:attribute:: YCC

        .. py:attribute:: RGB

as_image
--------

    .. py:function:: nvidia.nvimgcodecs.as_image(source_buffer: handle, cuda_stream: int = 0) -> nvimgcdcs::Image

        Wrap an external buffer as an image and tie the buffer lifetime to the image

        **Args:**

            ``source_buffer``: Object with __cuda_array_interface__.

            ``cuda_stream``: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place in created Image.

        **Returns:**

            ``nvidia.nvimgcodecs.Image``: Image which wraps provided source_buffer.

.. toctree::
    :caption: Python API
    :maxdepth: 2
    :hidden:
