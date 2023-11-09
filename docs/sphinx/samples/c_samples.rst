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

.. _all_c_samples:


C API samples
=============

This section will explain how to use nvImageCodec C API in ten quick steps to encode or decode image. The API details will be covered in the next section.

.. note:: All nvImageCodec APIs should return NVIMGCODEC_STATUS_SUCCESS. The results may not be valid otherwise.

Single image decode
-------------------

1. Initialize the library and creating library instance handle

.. code-block:: cpp

    nvimgcodecInstance_t instance;
    nvimgcodecInstanceCreateInfo_t instance_create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    instance_create_info.load_builtin_modules = 1;
    instance_create_info.load_extension_modules = 1;
    instance_create_info.create_debug_messenger = 1;

    nvimgcodecInstanceCreate(&instance, &instance_create_info);

2. Create code stream which abstracts the source of compressed bitstream. 
Lets start with creating :code:`nvimgcodecCodeStream_t` from file. 

.. code-block:: cpp

   nvimgcodecCodeStream_t code_stream;
   nvimgcodecCodeStreamCreateFromFile(instance, &code_stream, "./test.jpg")

3. Get image information about compressed image by getting  :code:`nvimgcodecImageInfo_t`` from Code Stream.
This operation will do minimal parsing of the compressed bitstream but without decoding it.

.. code-block:: cpp

    nvimgcodecImageInfo_t input_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecCodeStreamGetImageInfo(code_stream, &image_info);

    std::cout << "Input image info: " << std::endl;
    std::cout << "\t - width:" << input_image_info.plane_info[0].width << std::endl;
    std::cout << "\t - height:" << input_image_info.plane_info[0].height << std::endl;
    std::cout << "\t - components:" << input_image_info.num_planes << std::endl;
    std::cout << "\t - codec:" << input_image_info.codec_name << std::endl;

4. Allocate output memory on the device, and specify requested output image information.
The below snippet demonstrates initialization of :code:`nvimgcodecImageInfo_t` for an 8 bit,
SRGB, 3 channel image in planar format, without chroma subsampling and output buffer in strided device memory.

.. code-block:: cpp

   nvimgcodecImageInfo_t output_image_info(input_image_info);
   output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
   output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
   output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
   output_image_info.num_planes = 3;
   output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;

   auto sample_type = output_image_info.plane_info[0].sample_type;
   int bytes_per_element = static_cast<unsigned int>(sample_type)>> (8+3);
   size_t device_pitch_in_bytes = input_image_info.plane_info[0].width * bytes_per_element;

   for (uint32_t c = 0; c <  output_image_info.num_planes; ++c) {
      output_image_info.plane_info[c].height = input_image_info.plane_info[0].height;
      output_image_info.plane_info[c].width = input_image_info.plane_info[0].width;
      output_image_info.plane_info[c].row_stride = device_pitch_in_bytes;
   }

   output_image_info.buffer_size = output_image_info.plane_info[0].row_stride * image_info.plane_info[0].height * image_info.num_planes;
   output_image_info.cuda_stream = 0; // It is possible to assign cuda stream which will be used for synchronization. Here we assume it is default stream.

   cudaMalloc(&output_image_info.buffer,  output_image_info.buffer_size);

5. Having prepared requested output image format information, we can created opaque :code:`nvimgcodecImage_t` 
type handle to object which will represent our decoded image. 

.. code-block:: cpp

    nvimgcodecImage_t image;
    nvimgcodecImageCreate(instance, &image, &image_info);


6. Create Decoder. 

It is possible to pass Execution Parameters to create decoder function, where we can specify device id on which decoding will be executed. 
If we would like to pass some additional options to all or specific decoder plugin(s), we can do that by string with optional space 
separated list of parameters for specific decoders in format:  
"<decoder_id>:<parameter_name>=<parameter_value>". For example  "nvjpeg:fancy_upsampling=1". If we skip <decoder_id>, as in snippet bellow,
option can be interpreted by many decoder plugins.  

.. code-block:: cpp

    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
    nvimgcodecDecoder_t decoder;
    std::string dec_options{":fancy_upsampling=0"};
    nvimgcodecDecoderCreate(instance, &decoder, &exec_params, dec_options.c_str());

7. Prepare decoding parameters.
In below snippet we just set flag for allowing apply exif orientation.

.. code-block:: cpp

    nvimgcodecDecodeParams_t decode_params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
    decode_params.apply_exif_orientation = 1;


8. Schedule decoding.

.. code-block:: cpp

    nvimgcodecFuture_t decode_future;
    nvimgcodecDecoderDecode(decoder, &code_stream, &image, 1, &decode_params, &decode_future);

9. Wait for decoding to finish and check its status.

One of the fields in :code:`nvimgcodecImageInfo_t` is `cudaStream_t <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a>`__,
which the library uses to issue the asynchronous CUDA calls. To complete the decoding process, you need to call :code:`cudaDeviceSynchronize()` because 
:code:`nvimgcodecDecoderDecode` is asynchronous with respect to the host.
Alternatively, you can use  :code:`cudaStreamSynchronize` to synchronize with a specific CUDA stream.
Moreover, if you want to process the decoded image on the GPU, you can skip the synchronization here and use the CUDA stream defined in 
:code:`nvimgcodecImageInfo_t` to schedule further image processing on the GPU, which will occur after decoding.

.. code-block:: cpp

    size_t status_size;
    nvimgcodecProcessingStatus_t decode_status;
    nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
    cudaDeviceSynchronize(); // makes GPU wait until all decoding is finished
    if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during decoding - processing status: " << decode_status << std::endl;
    }
    nvimgcodecFutureDestroy(decode_future);

10. Cleanup

.. code-block:: cpp

    cudaFree(image_info.buffer);

    nvimgcodecImageDestroy(image);
    nvimgcodecDecoderDestroy(decoder);
    nvimgcodecCodeStreamDestroy(code_stream);
    nvimgcodecInstanceDestroy(instance);


Single image encode 
-------------------

The library expects the input image to be either on device or host memory and the compressed output will always be written to host memory.


1. Initialize the library and creating library instance handle

.. code-block:: cpp

    nvimgcodecInstance_t instance;
    nvimgcodecInstanceCreateInfo_t instance_create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    instance_create_info.load_builtin_modules = 1;
    instance_create_info.load_extension_modules = 1;
    instance_create_info.create_debug_messenger = 1;

    nvimgcodecInstanceCreate(&instance, &instance_create_info);

2. Prepare input image information.

The below snippet demonstrates initialization of  :code:`nvimgcodecImageInfo_t` with following assumptions:
 - sample type is unsigned 8 bits integer
 - sample format is planar RGB so 3 planes each with 1 channel (color component)
 - image is stored in host memory 
 
.. code-block:: cpp

    nvimgcodecImageInfo_t input_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};

    input_image_info->num_planes = 3;
    input_image_info->plane_info[0].height = /* TODO assign image height */;
    input_image_info->plane_info[0].width =  /* TODO assign image width */; 
    input_image_info->plane_info[0].num_channels = 1;


    input_image_info->color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    input_image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
    input_image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_444;

    auto sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8

    // For general case, having sample type,  we can calculate bytes per element using formula static_cast<unsigned int>(sample_type) >> (8 + 3);
    // so shift by 8 since 8..15 bits represents type bitdepth,  then shift by 3 to convert to # bytes 
    // here we can simple assign 1 as we assumed type is uint8
    int bytes_per_element =  1; 

    int pitch_in_bytes = input_image_info->plane_info[0].width * input_image_info->plane_info[0].num_channels * bytes_per_element;

    size_t buffer_size = 0;
    for (size_t p = 0; c < image_info->num_planes; c++) {
        input_image_info->plane_info[p].width = input_image_info->plane_info[0].width;
        input_image_info->plane_info[p].height = input_image_info->plane_info[0].height;
        input_image_info->plane_info[p].row_stride = pitch_in_bytes;
        input_image_info->plane_info[p].sample_type = sample_type;
        input_image_info->plane_info[p].num_channels = input_image_info->plane_info[0].num_channels; 
        input_image_info->plane_info[p].precision = 0;  //Value 0 means that precision is equal to sample type bitdepth (in our case 8 bits)
        buffer_size += input_image_info->plane_info[c].row_stride * input_image_info->plane_info[0].height;
    }

    input_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST; //or NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE if image is already in GPU memory
    input_image_info->buffer = /* TODO assign pointer to host memory where the image data in planar format are stored */;
    input_image_info->buffer_size = buffer_size;
    input_image_info.cuda_stream = 0; // It is possible to assign cuda stream which will be used for synchronization. Here we assume it is default stream.

3. Create opaque :code:`nvimgcodecImage_t` type handle to object which will represent our input image. 

.. code-block:: cpp

    nvimgcodecImage_t input_image;
    nvimgcodecImageCreate(instance, &input_image, &input_image_info);

4. Prepare output (compressed) image information. This information will be used in next step to create Code Stream 

Start with initializing based on input image information so we can have image size, sample type and other information already filled.
Than we can specify codec to encode with. In general, you can use codec names depending what extensions you have installed.
By default, in current version nvImageCodec comes with `jpeg` and `jpeg2k` gpu accelerated encoders. 
You can also use basic example CPU implementations of `bmp` and `pnm` encoders.
Here we choose `jpeg` codec.
We can also create structure for jpeg specific image information, where we can specify encoding as for example in below snippet choosing progressive dct Huffman.
To extend output (compressed) image information with jpeg specific information (here encoding type), we have to link jpeg image info structure to output image info.

.. code-block:: cpp

    nvimgcodecImageInfo_t out_image_info(input_image_info); 

    strcpy(out_image_info.codec_name,= "jpeg")

    nvimgcodecJpegImageInfo_t jpeg_image_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
    jpeg_image_info->encoding = NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;
    image_info.struct_next = &jpeg_image_info;

5. Create output code stream.

.. code-block:: cpp

    nvimgcodecCodeStream_t output_code_stream;
    nvimgcodecCodeStreamCreateToFile(instance, &output_code_stream, "out.jpg", &out_image_info)

6. Create encoder.

It is possible to pass Execution Parameters to create encoder function, where we can specify device id on which encoding will be executed. 
If we would like to pass some additional options to all or specific encoder plugin(s), we can do that by string with optional space 
separated list of parameters for specific encoders in format: "<encoder_id>:<parameter_name>=<parameter_value>".

.. code-block:: cpp

    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;

    nvimgcodecEncoder_t encoder;;
    nvimgcodecEncoderCreate(instance, &encoder, &exec_params, nullptr);

7. Prepare encode parameters.

For lossy compression we need to specify quality.

.. note:: For JPEG2000 we specify quality by providing target psnr e.g. :code:`encode_params.target_psnr = 35;`

.. code-block:: cpp

    nvimgcodecEncodeParams_t encode_params{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};

    encode_params.quality = 75;   

8. Schedule encoding.

.. code-block:: cpp

    nvimgcodecFuture_t encode_future;
    nvimgcodecEncoderEncode(encoder, &input_image, &output_code_stream, 1, &encode_params, &encode_future);
 
9. Wait for encoding to finish and read processing status.

.. code-block:: cpp
    
    size_t status_size;
    nvimgcodecProcessingStatus_t encode_status;
    nvimgcodecFutureGetProcessingStatus(encode_future, &encode_status, &status_size);
    cudaDeviceSynchronize(); // makes GPU wait until all encoding is finished
    encode_time = wtime() - encode_time;
    if (encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during encoding" << std::endl;
    }
    nvimgcodecFutureDestroy(encode_future);

10. Cleanup

.. code-block:: cpp

    nvimgcodecEncoderDestroy(decoder);
    nvimgcodecCodeStreamDestroy(code_stream);
    nvimgcodecImageDestroy(image);
    nvimgcodecInstanceDestroy(instance);

.. toctree::
    :maxdepth: 3
