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

C API
=====

.. toctree::
   :maxdepth: 3
   :hidden:

The nvImageCodec library and extension API.

Defines
-------

NVIMGCODEC_MAX_CODEC_NAME_SIZE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygendefine:: NVIMGCODEC_MAX_CODEC_NAME_SIZE

NVIMGCODEC_DEVICE_CURRENT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygendefine:: NVIMGCODEC_DEVICE_CURRENT

NVIMGCODEC_DEVICE_CPU_ONLY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygendefine:: NVIMGCODEC_DEVICE_CPU_ONLY

NVIMGCODEC_MAX_NUM_DIM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygendefine:: NVIMGCODEC_MAX_NUM_DIM

NVIMGCODEC_MAX_NUM_PLANES
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygendefine:: NVIMGCODEC_MAX_NUM_PLANES

NVIMGCODEC_JPEG2K_MAXRES
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygendefine:: NVIMGCODEC_JPEG2K_MAXRES


Typedefs
--------

nvimgcodecInstance_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecInstance_t

nvimgcodecImage_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecImage_t

nvimgcodecCodeStream_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecCodeStream_t

nvimgcodecParser_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecParser_t

nvimgcodecEncoder_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecEncoder_t

nvimgcodecDecoder_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecDecoder_t

nvimgcodecDebugMessenger_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecDebugMessenger_t

nvimgcodecExtension_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecExtension_t

nvimgcodecFuture_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecFuture_t

nvimgcodecDeviceMalloc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecDeviceMalloc_t

nvimgcodecDeviceFree_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecDeviceFree_t

nvimgcodecPinnedMalloc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecPinnedMalloc_t

nvimgcodecPinnedFree_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecPinnedFree_t

nvimgcodecProcessingStatus_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecProcessingStatus_t

nvimgcodecDebugCallback_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecDebugCallback_t

nvimgcodecLogFunc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecLogFunc_t

nvimgcodecExtensionModuleEntryFunc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecExtensionModuleEntryFunc_t

nvimgcodecResizeBufferFunc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: nvimgcodecResizeBufferFunc_t


Enums
-----

nvimgcodecStructureType_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecStructureType_t

nvimgcodecStatus_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecStatus_t

nvimgcodecSampleDataType_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecSampleDataType_t

nvimgcodecChromaSubsampling_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecChromaSubsampling_t

nvimgcodecSampleFormat_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecSampleFormat_t

nvimgcodecColorSpec_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecColorSpec_t

nvimgcodecImageBufferKind_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecImageBufferKind_t

nvimgcodecJpegEncoding_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecJpegEncoding_t

nvimgcodecBackendKind_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecBackendKind_t

nvimgcodecProcessingStatus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecProcessingStatus

nvimgcodecJpeg2kProgOrder_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecJpeg2kProgOrder_t

nvimgcodecJpeg2kBitstreamType_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecJpeg2kBitstreamType_t

nvimgcodecDebugMessageSeverity_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecDebugMessageSeverity_t

nvimgcodecDebugMessageCategory_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecDebugMessageCategory_t

nvimgcodecPriority_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: nvimgcodecPriority_t



Structures
----------

nvimgcodecBackend_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecBackend_t
  :members:

nvimgcodecBackendParams_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecBackendParams_t
  :members:

nvimgcodecCodeStreamDesc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecCodeStreamDesc_t
  :members:

nvimgcodecDebugMessageData_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecDebugMessageData_t
  :members:

nvimgcodecDebugMessengerDesc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecDebugMessengerDesc_t
  :members:

nvimgcodecDecodeParams_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecDecodeParams_t
  :members:

nvimgcodecDecoderDesc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecDecoderDesc_t
  :members:

nvimgcodecDeviceAllocator_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecDeviceAllocator_t
  :members:

nvimgcodecEncodeParams_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecEncodeParams_t
  :members:

nvimgcodecEncoderDesc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecEncoderDesc_t
  :members:

nvimgcodecExecutionParams_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecExecutionParams_t
  :members:

nvimgcodecExecutorDesc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecExecutorDesc_t
  :members:

nvimgcodecExtensionDesc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecExtensionDesc_t
  :members:

nvimgcodecFrameworkDesc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecFrameworkDesc_t
  :members:

nvimgcodecImageDesc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecImageDesc_t
  :members:

nvimgcodecImageInfo_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecImageInfo_t
  :members:

nvimgcodecImagePlaneInfo_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecImagePlaneInfo_t
  :members:

nvimgcodecInstanceCreateInfo_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecInstanceCreateInfo_t
  :members:

nvimgcodecIoStreamDesc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecIoStreamDesc_t
  :members:

nvimgcodecJpeg2kEncodeParams_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecJpeg2kEncodeParams_t
  :members:

nvimgcodecJpegEncodeParams_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecJpegEncodeParams_t
  :members:

nvimgcodecJpegImageInfo_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecJpegImageInfo_t
  :members:

nvimgcodecOrientation_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecOrientation_t
  :members:

nvimgcodecParserDesc_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecParserDesc_t
  :members:

nvimgcodecPinnedAllocator_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecPinnedAllocator_t
  :members:

nvimgcodecProperties_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecProperties_t
  :members:

nvimgcodecRegion_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: nvimgcodecRegion_t
  :members:


Functions
---------

nvimgcodecExtensionModuleEntry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecExtensionModuleEntry

nvimgcodecGetProperties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecGetProperties

nvimgcodecInstanceCreate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecInstanceCreate

nvimgcodecInstanceDestroy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecInstanceDestroy

nvimgcodecExtensionCreate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecExtensionCreate

nvimgcodecExtensionDestroy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecExtensionDestroy

nvimgcodecDebugMessengerCreate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecDebugMessengerCreate

nvimgcodecDebugMessengerDestroy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecDebugMessengerDestroy

nvimgcodecFutureWaitForAl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecFutureWaitForAll

nvimgcodecFutureDestroy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecFutureDestroy

nvimgcodecFutureGetProcessingStatus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecFutureGetProcessingStatus

nvimgcodecImageCreate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecImageCreate

nvimgcodecImageDestroy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecImageDestroy

nvimgcodecImageGetImageInfo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecImageGetImageInfo

nvimgcodecCodeStreamCreateFromFile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecCodeStreamCreateFromFile

nvimgcodecCodeStreamCreateFromHostMem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecCodeStreamCreateFromHostMem

nvimgcodecCodeStreamCreateToFile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecCodeStreamCreateToFile

nvimgcodecCodeStreamCreateToHostMem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecCodeStreamCreateToHostMem

nvimgcodecCodeStreamDestroy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecCodeStreamDestroy

nvimgcodecCodeStreamGetImageInfo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecCodeStreamGetImageInfo

nvimgcodecDecoderCreate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecDecoderCreate

nvimgcodecDecoderDestroy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecDecoderDestroy

nvimgcodecDecoderCanDecode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecDecoderCanDecode

nvimgcodecDecoderDecode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecDecoderDecode

nvimgcodecEncoderCreate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecEncoderCreate

nvimgcodecEncoderDestroy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecEncoderDestroy

nvimgcodecEncoderCanEncode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecEncoderCanEncode

nvimgcodecEncoderEncode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: nvimgcodecEncoderEncode
