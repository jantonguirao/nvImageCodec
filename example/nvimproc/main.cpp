/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <math.h>
#include <sys/types.h>
#include <filesystem>
#include <fstream>

#include <sys/stat.h>
#include <sys/types.h>

#include <dirent.h>

#include <cuda_runtime_api.h>
#include <cvcuda/OpCustomCrop.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpResize.hpp>

#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <nvcv_adapter.hpp>

#define COMMAND_PARAMS_IMPLEMENTATION
#include "command_line_params.h"

namespace fs = std::filesystem;

/**
 * @brief Crop and Resize sample app.
 *
 * The Crop and Resize is a simple pipeline which demonstrates usage of
 * CVCuda Tensor along with a few operators.
 *
 * Input Batch Tensor -> Crop -> Resize -> WriteImage
 */

inline void CheckCudaError(cudaError_t code, const char* file, const int line)
{
    if (code != cudaSuccess) {
        const char* errorMessage = cudaGetErrorString(code);
        const std::string message = "CUDA error returned at " + std::string(file) + ":" + std::to_string(line) +
                                    ", Error code: " + std::to_string(code) + " (" + std::string(errorMessage) + ")";
        throw std::runtime_error(message);
    }
}

#define CHECK_CUDA_ERROR(val)                      \
    {                                              \
        CheckCudaError((val), __FILE__, __LINE__); \
    }

double wtime(void)
{
#if defined(_WIN32)
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    } else {
        return (double)GetTickCount() / 1000.0;
    }
#else
    struct timespec tp;
    int rv = clock_gettime(CLOCK_MONOTONIC, &tp);

    if (rv)
        return 0;

    return tp.tv_nsec / 1.0E+9 + (double)tp.tv_sec;

#endif
}

uint32_t verbosity2severity(int verbose)
{
    uint32_t result = 0;
    if (verbose >= 1)
        result |= NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL | NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR;
    if (verbose >= 2)
        result |= NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING;
    if (verbose >= 3)
        result |= NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO;
    if (verbose >= 4)
        result |= NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG;
    if (verbose >= 5)
        result |= NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE;

    return result;
}

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char>> FileData;

int collect_input_files(const std::string& sInputPath, std::vector<std::string>& filelist)
{
    int error_code = 1;
    struct stat s;

    if (stat(sInputPath.c_str(), &s) == 0) {
        if (s.st_mode & S_IFREG) {
            filelist.push_back(sInputPath);
        } else if (s.st_mode & S_IFDIR) {
            // processing each file in directory
            DIR* dir_handle;
            struct dirent* dir;
            dir_handle = opendir(sInputPath.c_str());
            std::vector<std::string> filenames;
            if (dir_handle) {
                error_code = 0;
                while ((dir = readdir(dir_handle)) != NULL) {
                    if (dir->d_type == DT_REG) {
                        std::string sFileName = sInputPath + dir->d_name;
                        filelist.push_back(sFileName);
                    } else if (dir->d_type == DT_DIR) {
                        std::string sname = dir->d_name;
                        if (sname != "." && sname != "..") {
                            collect_input_files(sInputPath + sname + "/", filelist);
                        }
                    }
                }
                closedir(dir_handle);
            } else {
                std::cout << "Cannot open input directory: " << sInputPath << std::endl;
                return error_code;
            }
        } else {
            std::cout << "Cannot open input: " << sInputPath << std::endl;
            return error_code;
        }
    } else {
        std::cout << "Cannot find input path " << sInputPath << std::endl;
        return error_code;
    }

    return 0;
}

inline size_t sample_type_to_bytes_per_element(nvimgcdcsSampleDataType_t sample_type)
{
    return ((static_cast<unsigned int>(sample_type) & 0b11111110) + 7) / 8;
}

int decode_one_image(nvimgcdcsInstance_t instance, const CommandLineParams& params, const FileNames& image_names,
    nvimgcdcsSampleFormat_t out_format, NVCVImageData* image_data, NVCVTensorData* tensor_data, cudaStream_t& stream)
{
    int result = EXIT_SUCCESS;
    nvimgcdcsCodeStream_t code_stream;
    nvimgcdcsCodeStreamCreateFromFile(instance, &code_stream, image_names[0].c_str());
    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsCodeStreamGetImageInfo(code_stream, &image_info);
    char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
    nvimgcdcsCodeStreamGetCodecName(code_stream, codec_name);

    // Prepare decode parameters
    nvimgcdcsDecodeParams_t decode_params{};
    decode_params.enable_color_conversion = true;
    decode_params.enable_orientation = true;
    int bytes_per_element = sample_type_to_bytes_per_element(image_info.plane_info[0].sample_type);

    // Preparing output image_info
    image_info.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
    image_info.chroma_subsampling = NVIMGCDCS_SAMPLING_444;
    image_info.sample_format = out_format;
    if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB) {
        image_info.num_planes = 1;
        image_info.plane_info[0].num_channels = 3;
        image_info.plane_info[0].row_stride = image_info.plane_info[0].width * bytes_per_element * image_info.plane_info[0].num_channels;
        image_info.plane_info[0].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        image_info.buffer_size = image_info.plane_info[0].row_stride * image_info.plane_info[0].height * image_info.num_planes;
    } else if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) {
        size_t row_stride = image_info.plane_info[0].width * bytes_per_element;
        image_info.num_planes = 3;
        image_info.buffer_size = row_stride * image_info.plane_info[0].height * image_info.num_planes;
        for (auto p = 0; p < image_info.num_planes; ++p) {
            image_info.plane_info[p].height = image_info.plane_info[0].height;
            image_info.plane_info[p].width = image_info.plane_info[0].width;
            image_info.plane_info[p].row_stride = row_stride;
            image_info.plane_info[p].num_channels = 1;
            image_info.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        }
    }
    image_info.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE;

    //Allocate buffer
    CHECK_CUDA_ERROR(cudaMallocAsync(&image_info.buffer, image_info.buffer_size, stream));

    nvimgcdcsImage_t image;
    nvimgcdcsImageCreate(instance, &image, &image_info);

    nvimgcdcsDecoder_t decoder;
    nvimgcdcsDecoderCreate(instance, &decoder);

    nvimgcdcsFuture_t future;
    nvimgcdcsDecoderDecode(decoder, &code_stream, &image, 1, &decode_params, &future);

    nvimgcdcsProcessingStatus_t decode_status;
    size_t size;
    nvimgcdcsFutureGetProcessingStatus(future, &decode_status, &size);
    if (decode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during decoding" << std::endl;
        result = EXIT_FAILURE;
    }

    nvimgcdcsFutureDestroy(future);

    nvimgcdcsImageGetImageInfo(image, &image_info);
    nvimgcdcs::adapter::nvcv::ImageInfo2ImageData(image_data, image_info);
    nvimgcdcs::adapter::nvcv::ImageInfo2TensorData(tensor_data, image_info);

    nvimgcdcsImageDestroy(image);
    nvimgcdcsDecoderDestroy(decoder);
    nvimgcdcsCodeStreamDestroy(code_stream);

    return result;
}

void fill_encode_params(const CommandLineParams& params, fs::path output_path, nvimgcdcsEncodeParams_t* encode_params,
    nvimgcdcsJpeg2kEncodeParams_t* jpeg2k_encode_params, nvimgcdcsJpegEncodeParams_t* jpeg_encode_params)
{
    encode_params->type = NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS;
    encode_params->quality = params.quality;
    encode_params->target_psnr = params.target_psnr;
    encode_params->mct_mode = params.enc_color_trans ? NVIMGCDCS_MCT_MODE_YCC : NVIMGCDCS_MCT_MODE_RGB;

    //codec sepcific encode params
    if (params.output_codec == "jpeg2k") {
        jpeg2k_encode_params->type = NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
        jpeg2k_encode_params->stream_type =
            output_path.extension().string() == ".jp2" ? NVIMGCDCS_JPEG2K_STREAM_JP2 : NVIMGCDCS_JPEG2K_STREAM_J2K;
        jpeg2k_encode_params->code_block_w = params.code_block_w;
        jpeg2k_encode_params->code_block_h = params.code_block_h;
        jpeg2k_encode_params->irreversible = !params.reversible;
        jpeg2k_encode_params->prog_order = params.jpeg2k_prog_order;
        jpeg2k_encode_params->num_resolutions = params.num_decomps;
        encode_params->next = jpeg2k_encode_params;
    } else if (params.output_codec == "jpeg") {
        jpeg_encode_params->type = NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS;
        jpeg_encode_params->encoding = params.jpeg_encoding;
        jpeg_encode_params->optimized_huffman = params.optimized_huffman;
        encode_params->next = jpeg_encode_params;
    }
}

int encode_one_image(nvimgcdcsInstance_t instance, const CommandLineParams& params, const NVCVTensorData& tensor_data, fs::path output_path,
    cudaStream_t& stream)
{
    int result = EXIT_SUCCESS;

    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcs::adapter::nvcv::TensorData2ImageInfo(&image_info, tensor_data);
    if (0) {
        std::cout << "Input image info: " << std::endl;
        std::cout << "\t - width:" << image_info.plane_info[0].width << std::endl;
        std::cout << "\t - height:" << image_info.plane_info[0].height << std::endl;
        std::cout << "\t - num_planes:" << image_info.num_planes << std::endl;
        std::cout << "\t - num_channels:" << image_info.plane_info[0].num_channels << std::endl;
        std::cout << "\t - sample_format:" << image_info.sample_format << std::endl;
        std::cout << "\t - row_stride:" << image_info.plane_info[0].row_stride << std::endl;
        std::cout << "\t - codec:" << params.output_codec.data() << std::endl;
    }

    nvimgcdcsCodeStream_t code_stream;
    nvimgcdcsCodeStreamCreateToFile(instance, &code_stream, output_path.string().c_str(), params.output_codec.data(), &image_info);

    nvimgcdcsImage_t image;
    nvimgcdcsImageCreate(instance, &image, &image_info);

    nvimgcdcsEncodeParams_t encode_params{};
    nvimgcdcsJpeg2kEncodeParams_t jpeg2k_encode_params{};
    nvimgcdcsJpegEncodeParams_t jpeg_encode_params{};
    fill_encode_params(params, output_path, &encode_params, &jpeg2k_encode_params, &jpeg_encode_params);

    nvimgcdcsEncoder_t encoder;
    nvimgcdcsEncoderCreate(instance, &encoder);

    nvimgcdcsFuture_t future;
    nvimgcdcsEncoderEncode(encoder, &image, &code_stream, 1, &encode_params, &future);

    nvimgcdcsProcessingStatus_t encode_status;
    size_t status_size;
    nvimgcdcsFutureGetProcessingStatus(future, &encode_status, &status_size);
    if (encode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during encoding" << std::endl;
        result = EXIT_FAILURE;
    }
    nvimgcdcsFutureDestroy(future);
    nvimgcdcsEncoderDestroy(encoder);
    nvimgcdcsImageDestroy(image);
    nvimgcdcsCodeStreamDestroy(code_stream);

    return result;
}

int process_one_image(nvimgcdcsInstance_t instance, fs::path input_path, fs::path output_path, const CommandLineParams& params)
{
    // tag: Scan input directory and collect source images
    FileNames image_names;
    collect_input_files(input_path.string(), image_names);

    // tag: Create the cuda stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // tag: Image Loading
    // nvImageCodecs is used to load the image

    NVCVImageData image_data;
    NVCVTensorData tensor_data;
    decode_one_image(instance, params, image_names, NVIMGCDCS_SAMPLEFORMAT_P_RGB, &image_data, &tensor_data, stream);
    if (0) { // Example for image creation
        nvcv::ImageDataStridedCuda imageDataStridedCuda(image_data);
        nvcv::ImageWrapData inImage(imageDataStridedCuda);
        nvcv::TensorWrapImage decodedTensor(inImage);
    }

    nvcv::TensorDataStridedCuda tensorDataStridedCuda(tensor_data);
    nvcv::TensorWrapData decodedTensor(tensorDataStridedCuda);

    // tag: Create operator input tensor in RGB interleaved format
    nvcv::Tensor inTensor(1, {static_cast<int>(tensor_data.shape[3]), static_cast<int>(tensor_data.shape[2])}, nvcv::FMT_RGB8);

    // tag: Convert form planar to interleaved 
    cvcuda::Reformat reformatOp;
    reformatOp(stream, decodedTensor, inTensor);
  
    // tag: The input buffer is now ready to be used by the operators

    // Set parameters for Crop and Resize
    // ROI dimensions to crop in the input image
    int cropX = 50;
    int cropY = 100;
    int cropWidth = 320;
    int cropHeight = 240;

    // Set the resize dimensions
    int resizeWidth = 800;
    int resizeHeight = 600;

    //  Initialize the CVCUDA ROI struct
    NVCVRectI crpRect = {cropX, cropY, cropWidth, cropHeight};

    // tag: Allocate Tensors for Crop and Resize
    
    // Create a CVCUDA Tensor based on the crop window size.
    nvcv::Tensor cropTensor(1, {cropWidth, cropHeight}, nvcv::FMT_RGB8);
    // Create a CVCUDA Tensor based on resize dimensions
    nvcv::Tensor resizedTensor(1, {resizeWidth, resizeHeight}, nvcv::FMT_RGB8);

#ifdef PROFILE_SAMPLE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    // tag: Initialize operators for Crop and Resize
    cvcuda::CustomCrop cropOp;
    cvcuda::Resize resizeOp;

    // tag: Executes the CustomCrop operation on the given cuda stream
    cropOp(stream, inTensor, cropTensor, crpRect);
    //cropOp(stream, decodedTensor, cropTensor, crpRect);

    // Resize operator can now be enqueued into the same stream
    resizeOp(stream, cropTensor, resizedTensor, NVCV_INTERP_LINEAR);

    // tag: Profile section
#ifdef PROFILE_SAMPLE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float operatorms = 0;
    cudaEventElapsedTime(&operatorms, start, stop);
    std::cout << "Time for Crop and Resize : " << operatorms << " ms" << std::endl;
#endif

    // tag: Create output tensor in planar RGB as currently all codecs supports this format
    nvcv::Tensor outTensor(1, {resizeWidth, resizeHeight}, nvcv::FMT_RGB8p);

    // tag: Reformat interleaved to planar
    reformatOp(stream, resizedTensor, outTensor);

    // tag: Copy the buffer to CPU and write resized image into output file
    const NVCVTensorData& out_tensor_data = outTensor.exportData()->cdata();
    encode_one_image(instance, params, out_tensor_data, output_path, stream);

    // tag: Clean up
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}

void list_cuda_devices()
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}

int main(int argc, const char* argv[])
{
    int exit_code = EXIT_SUCCESS;
    CommandLineParams params;
    int status = process_commandline_params(argc, argv, &params);
    if (status != -1) {
        return status;
    }

    if (params.list_cuda_devices) {
        list_cuda_devices();
    }

    nvimgcdcsInstance_t instance;
    nvimgcdcsInstanceCreateInfo_t instance_create_info{};
    instance_create_info.type = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.next = NULL;
    instance_create_info.pinned_allocator = NULL;
    instance_create_info.device_allocator = NULL;
    instance_create_info.load_extension_modules = true;
    instance_create_info.default_debug_messenger = true;
    instance_create_info.message_severity = verbosity2severity(params.verbose);
    instance_create_info.message_type = NVIMGCDCS_DEBUG_MESSAGE_TYPE_ALL;
    instance_create_info.num_cpu_threads = 10;

    nvimgcdcsInstanceCreate(&instance, instance_create_info);

    fs::path exe_path(argv[0]);
    fs::path input_path = fs::absolute(exe_path).parent_path() / fs::path(params.input);
    fs::path output_path = fs::absolute(exe_path).parent_path() / fs::path(params.output);

    if (fs::is_directory(input_path)) {
        // exit_code = process_images(instance, input_path, output_path, params);
    } else {
        exit_code = process_one_image(instance, input_path, output_path, params);
    }
    nvimgcdcsInstanceDestroy(instance);

    return exit_code;
}
