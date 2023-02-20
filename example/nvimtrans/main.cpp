#include <cuda_runtime_api.h>
#include <nvimgcodecs.h>
#include <stdlib.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "command_line_params.h"
#include "nvpxm.h"

namespace fs = std::filesystem;

#if defined(_WIN32)
    #include <windows.h>
#endif

#define CHECK_CUDA(call)                                                                \
    {                                                                                   \
        cudaError_t _e = (call);                                                        \
        if (_e != cudaSuccess) {                                                        \
            std::cerr << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                         \
            return EXIT_FAILURE;                                                        \
        }                                                                               \
    }

#define CHECK_NVIMGCDCS(call)                                   \
    {                                                           \
        nvimgcdcsStatus_t _e = (call);                          \
        if (_e != NVIMGCDCS_STATUS_SUCCESS) {                   \
            std::stringstream _error;                           \
            _error << "nvImageCodecs failure: '#" << _e << "'"; \
            throw std::runtime_error(_error.str());             \
        }                                                       \
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

void fill_encode_params(const CommandLineParams& params, fs::path output_path,
    nvimgcdcsEncodeParams_t* encode_params, nvimgcdcsJpeg2kEncodeParams_t* jpeg2k_encode_params,
    nvimgcdcsJpegEncodeParams_t* jpeg_encode_params)
{
    memset(encode_params, 0, sizeof(nvimgcdcsEncodeParams_t));
    encode_params->type = NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS;
    encode_params->quality = params.quality;
    encode_params->target_psnr = params.target_psnr;
    encode_params->mct_mode =
        params.enc_color_trans ? NVIMGCDCS_MCT_MODE_YCC : NVIMGCDCS_MCT_MODE_RGB;

    //codec sepcific encode params
    if (params.output_codec == "jpeg2k") {
        memset(jpeg2k_encode_params, 0, sizeof(nvimgcdcsJpeg2kEncodeParams_t));
        jpeg2k_encode_params->type = NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
        jpeg2k_encode_params->stream_type = output_path.extension().string() == ".jp2"
                                                ? NVIMGCDCS_JPEG2K_STREAM_JP2
                                                : NVIMGCDCS_JPEG2K_STREAM_J2K;
        jpeg2k_encode_params->code_block_w = params.code_block_w;
        jpeg2k_encode_params->code_block_h = params.code_block_h;
        jpeg2k_encode_params->irreversible = !params.reversible;
        jpeg2k_encode_params->prog_order = NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL;
        jpeg2k_encode_params->num_resolutions = params.num_decomps;

        //TODO Support for more jpeg2k specific parameters
        // uint16_t rsiz;
        // uint32_t enable_SOP_marker;
        // uint32_t enable_EPH_marker;
        // nvimgcdcsJpeg2kProgOrder_t prog_order;
        // uint32_t num_layers;
        // uint32_t encode_modes;
        // uint32_t enable_custom_precincts;
        // uint32_t precint_width[NVIMGCDCS_JPEG2K_MAXRES];
        // uint32_t precint_height[NVIMGCDCS_JPEG2K_MAXRES];

        encode_params->next = jpeg2k_encode_params;
    } else if (params.output_codec == "jpeg") {
        memset(jpeg_encode_params, 0, sizeof(nvimgcdcsJpegEncodeParams_t));
        jpeg_encode_params->type = NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS;
        jpeg_encode_params->encoding = params.jpeg_encoding;
        jpeg_encode_params->optimized_huffman = params.optimized_huffman;
        encode_params->next = jpeg_encode_params;
    }
}

int process_one_image(nvimgcdcsInstance_t instance, fs::path input_path, fs::path output_path,
    const CommandLineParams& params)
{
    double total_time = 0.;
    double parse_time = 0.;
    double decode_time = 0.;
    int total_images = 1;
    nvimgcdcsCodeStream_t code_stream;

    std::cout << "Loading " << input_path.string() << " file" << std::endl;
    std::ifstream file(input_path.string(), std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
    if (0) {
        nvimgcdcsCodeStreamCreateFromFile(instance, &code_stream, input_path.string().c_str());
    } else {
        nvimgcdcsCodeStreamCreateFromHostMem(instance, &code_stream, buffer.data(), buffer.size());
    }

    parse_time = wtime();
    nvimgcdcsImageInfo_t image_info;
    nvimgcdcsCodeStreamGetImageInfo(code_stream, &image_info);
    parse_time = wtime() - parse_time;
    char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
    nvimgcdcsCodeStreamGetCodecName(code_stream, codec_name);

    std::cout << "Input image info: " << std::endl;
    std::cout << "\t - width:" << image_info.image_width << std::endl;
    std::cout << "\t - height:" << image_info.image_height << std::endl;
    std::cout << "\t - components:" << image_info.num_components << std::endl;
    std::cout << "\t - codec:" << codec_name << std::endl;

    nvimgcdcsDecodeParams_t decode_params;
    memset(&decode_params, 0, sizeof(nvimgcdcsDecodeParams_t));
    decode_params.enable_color_conversion = params.dec_color_trans;
    decode_params.enable_orientation = !params.ignore_orientation;
    if (decode_params.enable_orientation) {
        decode_params.orientation.rotated = image_info.orientation.rotated == 90
                                                ? 270
                                                : (image_info.orientation.rotated == 270 ? 90 : 0);
        if (decode_params.orientation.rotated) {
            auto tmp = image_info.image_width;
            image_info.image_width = image_info.image_height;
            image_info.image_height = tmp;
        }
    }
    int bytes_per_element = image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 1 : 2;
    image_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
    image_info.color_space = NVIMGCDCS_COLORSPACE_SRGB;

    nvimgcdcsImage_t image;
    nvimgcdcsImageCreate(instance, &image);

    nvimgcdcsDecoder_t decoder;
    nvimgcdcsDecoderCreate(instance, &decoder, code_stream, image, &decode_params);

    nvimgcdcsDecodeState_t decode_state;
    nvimgcdcsDecodeStateCreate(decoder, &decode_state, nullptr);

    size_t capabilities_size;
    nvimgcdcsDecoderGetCapabilities(decoder, nullptr, &capabilities_size);
    const nvimgcdcsCapability_t* capabilities_ptr;
    nvimgcdcsDecoderGetCapabilities(decoder, &capabilities_ptr, &capabilities_size);

    bool is_host_output = std::find(capabilities_ptr,
                              capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                              NVIMGCDCS_CAPABILITY_HOST_OUTPUT) !=
                          capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
    bool is_device_output =
        std::find(capabilities_ptr,
            capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
            NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT) !=
        capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);

    std::cout << "Saving to " << output_path.string() << " file" << std::endl;

    nvimgcdcsCodeStream_t output_code_stream;
    nvimgcdcsCodeStreamCreateToFile(
        instance, &output_code_stream, output_path.string().c_str(), params.output_codec.data());
    nvimgcdcsImageInfo_t out_image_info(image_info);
    out_image_info.sampling = params.chroma_subsampling;
    nvimgcdcsCodeStreamSetImageInfo(output_code_stream, &out_image_info);

    nvimgcdcsEncodeParams_t encode_params;
    memset(&encode_params, 0, sizeof(nvimgcdcsEncodeParams_t));
    nvimgcdcsJpeg2kEncodeParams_t jpeg2k_encode_params;
    nvimgcdcsJpegEncodeParams_t jpeg_encode_params;
    fill_encode_params(
        params, output_path, &encode_params, &jpeg2k_encode_params, &jpeg_encode_params);

    nvimgcdcsImageSetImageInfo(image, &image_info);
    nvimgcdcsEncoder_t encoder = nullptr;
    nvimgcdcsEncoderCreate(instance, &encoder, image, output_code_stream, &encode_params);

    nvimgcdcsEncoderGetCapabilities(encoder, nullptr, &capabilities_size);
    nvimgcdcsEncoderGetCapabilities(encoder, &capabilities_ptr, &capabilities_size);

    bool is_host_input = std::find(capabilities_ptr,
                             capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                             NVIMGCDCS_CAPABILITY_HOST_INPUT) !=
                         capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
    bool is_device_input = std::find(capabilities_ptr,
                               capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                               NVIMGCDCS_CAPABILITY_DEVICE_INPUT) !=
                           capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);

    bool is_interleaved = static_cast<int>(image_info.sample_format) % 2 == 0;

    unsigned char* device_buffer = nullptr;
    std::vector<unsigned char> host_buffer;

    if (is_host_output || is_host_input) {
        host_buffer.resize(image_info.image_width * image_info.image_height *
                           image_info.num_components * bytes_per_element);
        for (auto c = 0; c < image_info.num_components; ++c) {
            image_info.component_info[c].host_pitch_in_bytes =
                image_info.image_width * (is_interleaved ? image_info.num_components : 1);
        }
        nvimgcdcsImageSetHostBuffer(image, host_buffer.data(), host_buffer.size());
    }
    if (is_device_output || is_device_input) {
        size_t device_pitch_in_bytes = 0;
        CHECK_CUDA(cudaMallocPitch((void**)&device_buffer, &device_pitch_in_bytes,
            image_info.image_width * bytes_per_element *
                (is_interleaved ? image_info.num_components : 1),
            image_info.image_height * (is_interleaved ? 1 : image_info.num_components)));
        for (auto c = 0; c < image_info.num_components; ++c) {
            image_info.component_info[c].device_pitch_in_bytes = device_pitch_in_bytes;
        }
        size_t image_buffer_size =
            device_pitch_in_bytes * image_info.image_height * image_info.num_components;
        nvimgcdcsImageSetDeviceBuffer(image, device_buffer, image_buffer_size);
    }

    nvimgcdcsImageSetImageInfo(image, &image_info);
    nvimgcdcsImageAttachDecodeState(image, decode_state);

    // warm up
    for (int warmup_iter = 0; warmup_iter < params.warmup; warmup_iter++) {
        if (warmup_iter == 0) {
            std::cout << "Warmup started!" << std::endl;
        }
        nvimgcdcsDecoderDecode(decoder, code_stream, image, &decode_params, nullptr, true);
        nvimgcdcsProcessingStatus_t decode_status;
        nvimgcdcsImageGetProcessingStatus(image, &decode_status);
        if (decode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
            std::cerr << "Error: Something went wrong during warmup decoding" << std::endl;
        }
        if (warmup_iter == (params.warmup - 1)) {
            std::cout << "Warmup done!" << std::endl;
        }
    }
    decode_time = wtime();
    nvimgcdcsDecoderDecode(decoder, code_stream, image, &decode_params, nullptr, true);

    nvimgcdcsProcessingStatus_t decode_status;
    nvimgcdcsImageGetProcessingStatus(image, &decode_status);
    if (decode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during decoding" << std::endl;
    }

    decode_time = wtime() - decode_time;

    // assert(ready_decoded_image == image); //we sent only one image to decoder

    total_time = parse_time + decode_time;
    std::cout << "Total images processed: " << total_images << std::endl;
    std::cout << "Total time spent on decoding: " << total_time << std::endl;
    std::cout << "Avg time/image: " << total_time / total_images << std::endl;

    if (is_host_output && is_device_input) {
        CHECK_CUDA(cudaMemcpy2D(device_buffer,
            (size_t)image_info.component_info[0].device_pitch_in_bytes, host_buffer.data(),
            (size_t)image_info.component_info[0].host_pitch_in_bytes, image_info.image_width,
            image_info.image_height * (is_interleaved ? 1 : image_info.num_components),
            cudaMemcpyHostToDevice));
    } else if (is_device_output && is_host_input) {
        CHECK_CUDA(cudaMemcpy2D(host_buffer.data(),
            (size_t)image_info.component_info[0].host_pitch_in_bytes, device_buffer,
            (size_t)image_info.component_info[0].device_pitch_in_bytes,
            (size_t)image_info.component_info[0].host_pitch_in_bytes,
            image_info.image_height * (is_interleaved ? 1 : image_info.num_components),
            cudaMemcpyDeviceToHost));
        nvimgcdcsImageSetImageInfo(image, &image_info);
    }

    nvimgcdcsEncodeState_t encode_state;
    nvimgcdcsEncodeStateCreate(encoder, &encode_state, nullptr);
    nvimgcdcsImageAttachEncodeState(image, encode_state);
    nvimgcdcsEncoderEncode(encoder, output_code_stream, image, &encode_params, nullptr, true);

    nvimgcdcsProcessingStatus_t encode_status;
    nvimgcdcsImageGetProcessingStatus(image, &encode_status);
    if (decode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during encoding" << std::endl;
    }

    nvimgcdcsImageDetachEncodeState(image);
    nvimgcdcsImageDetachDecodeState(image);

    nvimgcdcsEncodeStateDestroy(encode_state);
    nvimgcdcsEncoderDestroy(encoder);
    nvimgcdcsCodeStreamDestroy(output_code_stream);

    if (device_buffer) {
        CHECK_CUDA(cudaFree(device_buffer));
    }
    nvimgcdcsImageDestroy(image);
    nvimgcdcsDecodeStateDestroy(decode_state);
    nvimgcdcsDecoderDestroy(decoder);
    nvimgcdcsCodeStreamDestroy(code_stream);

    return EXIT_SUCCESS;
}

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char>> FileData;

int collect_input_files(const std::string& sInputPath, std::vector<std::string>& filelist)
{

    if (fs::is_regular_file(sInputPath)) {
        filelist.push_back(sInputPath);
    } else if (fs::is_directory(sInputPath)) {
        fs::recursive_directory_iterator iter(sInputPath);
        for (auto& p : iter) {
            if (fs::is_regular_file(p)) {
                filelist.push_back(p.path().string());
            }
        }
    } else {
        std::cout << "unable to open input" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int read_next_batch(FileNames& image_names, int batch_size, FileNames::iterator& cur_iter,
    FileData& raw_data, std::vector<size_t>& raw_len, FileNames& current_names, bool verbose)
{
    int counter = 0;

    while (counter < batch_size) {
        if (cur_iter == image_names.end()) {
            if (verbose) {
                std::cerr << "Image list is too short to fill the batch, adding files "
                             "from the beginning of the image list"
                          << std::endl;
            }
            cur_iter = image_names.begin();
        }

        if (image_names.size() == 0) {
            std::cerr << "No valid images left in the input list, exit" << std::endl;
            return EXIT_FAILURE;
        }

        // Read an image from disk.
        std::ifstream input(cur_iter->c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open())) {
            std::cerr << "Cannot open image: " << *cur_iter << ", removing it from image list"
                      << std::endl;
            image_names.erase(cur_iter);
            continue;
        }

        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (raw_data[counter].size() < static_cast<size_t>(file_size)) {
            raw_data[counter].resize(file_size);
        }
        if (!input.read(raw_data[counter].data(), file_size)) {
            std::cerr << "Cannot read from file: " << *cur_iter << ", removing it from image list"
                      << std::endl;
            image_names.erase(cur_iter);
            continue;
        }
        raw_len[counter] = file_size;

        current_names[counter] = *cur_iter;

        counter++;
        cur_iter++;
    }
    return EXIT_SUCCESS;
}

constexpr int NUM_COMPONENTS = 4;
struct ImageBuffer
{
    unsigned char* data;
    size_t size;
    size_t pitch_in_bytes;
};

int prepare_decode_resources(nvimgcdcsInstance_t instance, FileData& file_data,
    std::vector<size_t>& file_len, std::vector<ImageBuffer>& ibuf, FileNames& current_names,
    nvimgcdcsDecoder_t& decoder, nvimgcdcsDecodeState_t& decode_state_batch, bool& is_host_output,
    bool& is_device_output, std::vector<nvimgcdcsCodeStream_t>& code_streams,
    std::vector<nvimgcdcsImage_t>& images, std::vector<nvimgcdcsDecodeState_t>& decode_states,
    const nvimgcdcsDecodeParams_t& decode_params, double& parse_time)
{
    parse_time = 0;
    for (uint32_t i = 0; i < file_data.size(); i++) {
        CHECK_NVIMGCDCS(nvimgcdcsCodeStreamCreateFromHostMem(
            instance, &code_streams[i], (unsigned char*)file_data[i].data(), file_len[i]));

        double time = wtime();
        nvimgcdcsImageInfo_t image_info;
        CHECK_NVIMGCDCS(nvimgcdcsCodeStreamGetImageInfo(code_streams[i], &image_info));
        parse_time += wtime() - time;

        if (image_info.num_components > NUM_COMPONENTS) {
            std::cout << "Num Components > " << NUM_COMPONENTS << "not supported by this sample"
                      << std::endl;
            return EXIT_FAILURE;
        }

        for (uint32_t c = 0; c < image_info.num_components; ++c) {
            if (image_info.component_info[c].sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) {
                std::cout << "Precision not supported by this sample" << std::endl;
                return EXIT_FAILURE;
            }
        }

        char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
        CHECK_NVIMGCDCS(nvimgcdcsCodeStreamGetCodecName(code_streams[i], codec_name));
        // std::cout << "Input image info: " << std::endl;
        // std::cout << "\t - width:" << image_info.image_width << std::endl;
        // std::cout << "\t - height:" << image_info.image_height << std::endl;
        // std::cout << "\t - components:" << image_info.num_components << std::endl;
        // std::cout << "\t - codec:" << codec_name << std::endl;

        if (images[i] == nullptr) {
            CHECK_NVIMGCDCS(nvimgcdcsImageCreate(instance, &images[i]));
        }
        //TODO orientation
        /* 
        if (decode_params.enable_orientation) {
            decode_params.orientation.rotated =
                image_info.orientation.rotated == 90
                    ? 270
                    : (image_info.orientation.rotated == 270 ? 90 : 0);
            if (decode_params.orientation.rotated) {
                auto tmp                = image_info.image_width;
                image_info.image_width  = image_info.image_height;
                image_info.image_height = tmp;
            }
        }
        */
        int bytes_per_element = image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 1 : 2;
        //Decode to format
        image_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        image_info.color_space = NVIMGCDCS_COLORSPACE_SRGB;

        // realloc output buffer if required
        // we are allocating assuming that all component dimensions are the same

        /*
        std::vector<unsigned char> host_buffer;

        if (is_host_output || is_host_input) {
            host_buffer.resize(image_info.image_width * image_info.image_height *
                                image_info.num_components * bytes_per_element);
            image_info.component_info[0].host_pitch_in_bytes =
                image_info.image_width * (is_interleaved ? image_info.num_components : 1);
            image_info.component_info[1].host_pitch_in_bytes =
                image_info.image_width * (is_interleaved ? image_info.num_components : 1);
            image_info.component_info[2].host_pitch_in_bytes =
                image_info.image_width * (is_interleaved ? image_info.num_components : 1);

            nvimgcdcsImageSetHostBuffer(image, host_buffer.data(), host_buffer.size());
        }
        */

        CHECK_NVIMGCDCS(nvimgcdcsImageSetImageInfo(images[i], &image_info));

        if (decoder == nullptr) {
            if (0) {
                CHECK_NVIMGCDCS(nvimgcdcsDecoderCreate(
                    instance, &decoder, code_streams[i], images[i], &decode_params));
                CHECK_NVIMGCDCS(
                    nvimgcdcsDecodeStateBatchCreate(decoder, &decode_state_batch, nullptr));
            } else {
                CHECK_NVIMGCDCS(nvimgcdcsDecoderGenericCreate(instance, &decoder));
                CHECK_NVIMGCDCS(
                    nvimgcdcsDecodeStateBatchCreate(decoder, &decode_state_batch, nullptr));
            }
         
            size_t capabilities_size;
            CHECK_NVIMGCDCS(nvimgcdcsDecoderGetCapabilities(decoder, nullptr, &capabilities_size));
            const nvimgcdcsCapability_t* capabilities_ptr;
            CHECK_NVIMGCDCS(
                nvimgcdcsDecoderGetCapabilities(decoder, &capabilities_ptr, &capabilities_size));
            // bool can_batch =
            //     std::find(capabilities_ptr, 
            //         capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
            //         NVIMGCDCS_CAPABILITY_BATCH) !=
            //     capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
            // if (!can_batch) {
            //     std::cerr << "decoder cannot batch processing" << std::endl;
            // }
            is_host_output =
                std::find(capabilities_ptr,
                    capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                    NVIMGCDCS_CAPABILITY_HOST_OUTPUT) !=
                capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
            is_device_output =
                std::find(capabilities_ptr,
                    capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                    NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT) !=
                capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
        } else {
            //std::cout << "Check if decoder can decode image " << std::endl;
            bool can_decode;
            CHECK_NVIMGCDCS(nvimgcdcsDecoderCanDecode(
                decoder, &can_decode, code_streams[i], images[i], &decode_params));
            if (!can_decode) {
                std::cerr << "skipping this image since it can be decoded in this batch"
                          << std::endl;
            }
        }
        
        if (is_device_output /*|| is_device_input*/) {
            size_t device_pitch_in_bytes = bytes_per_element * image_info.image_width;
            size_t image_buffer_size =
                device_pitch_in_bytes * image_info.image_height * image_info.num_components;

            if (image_buffer_size > ibuf[i].size) {
                if (ibuf[i].data) {
                    CHECK_CUDA(cudaFree(ibuf[i].data));
                }
                CHECK_CUDA(cudaMalloc((void**)&ibuf[i].data, image_buffer_size));
            }
            ibuf[i].pitch_in_bytes = device_pitch_in_bytes;
            ibuf[i].size = image_buffer_size;

            for (uint32_t c = 0; c < image_info.num_components; ++c) {
                image_info.component_info[c].device_pitch_in_bytes = device_pitch_in_bytes;
            }
            unsigned char* device_buffer = ibuf[i].data;
            CHECK_NVIMGCDCS(
                nvimgcdcsImageSetDeviceBuffer(images[i], device_buffer, image_buffer_size));
        }
        CHECK_NVIMGCDCS(nvimgcdcsImageSetImageInfo(images[i], &image_info));

        if (decode_states[i] == nullptr) {
            CHECK_NVIMGCDCS(nvimgcdcsDecodeStateCreate(decoder, &decode_states[i], nullptr));
            CHECK_NVIMGCDCS(nvimgcdcsImageAttachDecodeState(images[i], decode_states[i]));
        }
    }
    return EXIT_SUCCESS;
}

static std::map<std::string, std::string> codec2ext = {{"bmp", ".bmp"}, {"jpeg2k", ".j2k"},
    {"tiff", ".tiff"}, {"jpeg", ".jpg"}, {"pxm", ".ppm"}};

int prepare_encode_resources(nvimgcdcsInstance_t instance, FileNames& current_names,
    nvimgcdcsEncoder_t& encoder, nvimgcdcsEncodeState_t& encode_state_batch, bool& is_host_input,
    bool& is_device_input, std::vector<nvimgcdcsCodeStream_t>& out_code_streams,
    std::vector<nvimgcdcsImage_t>& images, std::vector<nvimgcdcsEncodeState_t>& encode_states,
    const nvimgcdcsEncodeParams_t& encode_params,
    const CommandLineParams& params, fs::path output_path)
{
    for (uint32_t i = 0; i < current_names.size(); i++) {
        fs::path filename = fs::path(current_names[i]).filename();
        //TODO extension can depend on image format
        std::string ext = "___";
        auto it = codec2ext.find(params.output_codec);
        if (it != codec2ext.end()) {
            ext = it->second;
        }

        filename = filename.replace_extension(ext); 
        fs::path output_filename = output_path / filename;

        CHECK_NVIMGCDCS(nvimgcdcsCodeStreamCreateToFile(
            instance, &out_code_streams[i], output_filename.string().c_str(), params.output_codec.c_str()));
        nvimgcdcsImageInfo_t image_info;
        nvimgcdcsImageGetImageInfo(images[i], &image_info);
        nvimgcdcsImageInfo_t out_image_info(image_info);
        out_image_info.sampling = params.chroma_subsampling;
        nvimgcdcsCodeStreamSetImageInfo(out_code_streams[i], &out_image_info);

        if (encoder == nullptr) {
            CHECK_NVIMGCDCS(nvimgcdcsEncoderCreate(
                instance, &encoder, images[i], out_code_streams[i], &encode_params))
            CHECK_NVIMGCDCS(nvimgcdcsEncodeStateBatchCreate(encoder, &encode_state_batch, nullptr));
        }
        if (encode_states[i] == nullptr) {
            CHECK_NVIMGCDCS(nvimgcdcsEncodeStateCreate(encoder, &encode_states[i], NULL));
        }
        CHECK_NVIMGCDCS(nvimgcdcsImageAttachEncodeState(images[i], encode_states[i]));
    }
    return EXIT_SUCCESS;
}

int process_images(nvimgcdcsInstance_t instance, fs::path input_path, fs::path output_path,
    const CommandLineParams& params)
{
    double time = wtime();
    FileNames image_names;
    collect_input_files(input_path.string(), image_names);
    int total_images = params.total_images;
    if (total_images == -1) {
        total_images = image_names.size();
    }
    // vector for storing raw files and file lengths
    FileData file_data(params.batch_size);
    std::vector<size_t> file_len(params.batch_size);

    FileNames current_names(params.batch_size);
    // we wrap over image files to process total_images of files
    FileNames::iterator file_iter = image_names.begin();

    std::vector<ImageBuffer> image_buffers(params.batch_size);

    nvimgcdcsDecoder_t decoder = nullptr;
    nvimgcdcsEncoder_t encoder = nullptr;

    nvimgcdcsDecodeState_t decode_state_batch = nullptr;
    nvimgcdcsEncodeState_t encode_state_batch = nullptr;

    std::vector<nvimgcdcsCodeStream_t> in_code_streams(params.batch_size);
    std::vector<nvimgcdcsCodeStream_t> out_code_streams(params.batch_size);
    std::vector<nvimgcdcsImage_t> images(params.batch_size);
    std::vector<nvimgcdcsDecodeState_t> decode_states(params.batch_size);
    std::vector<nvimgcdcsEncodeState_t> encode_states(params.batch_size);

    nvimgcdcsDecodeParams_t decode_params;
    memset(&decode_params, 0, sizeof(nvimgcdcsDecodeParams_t));
    decode_params.enable_color_conversion = params.dec_color_trans;
    decode_params.enable_orientation = !params.ignore_orientation;

    nvimgcdcsEncodeParams_t encode_params;
    nvimgcdcsJpeg2kEncodeParams_t jpeg2k_encode_params;
    nvimgcdcsJpegEncodeParams_t jpeg_encode_params;
    fill_encode_params(
        params, output_path, &encode_params, &jpeg2k_encode_params, &jpeg_encode_params);

    bool is_host_output = false;
    bool is_device_output = false;
    bool is_host_input = false;
    bool is_device_input = false;

    cudaEvent_t startEvent = NULL;
    cudaEvent_t stopEvent = NULL;

    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

    float loopTime = 0;

    int total_processed = 0;

    double total_processing_time = 0;
    double total_reading_time = 0;
    double total_parse_time = 0;
    double total_decode_time = 0;
    double total_encode_time = 0;
    double total_time = 0;

    int warmup = 0;
    while (total_processed < total_images) {
        double start_reading_time = wtime();
        if (read_next_batch(image_names, params.batch_size, file_iter, file_data, file_len,
                current_names, params.verbose))
            return EXIT_FAILURE;
        double reading_time = wtime() - start_reading_time;

        double parse_time = 0;
        if (prepare_decode_resources(instance, file_data, file_len, image_buffers, current_names,
                decoder, decode_state_batch, is_host_output, is_device_output, in_code_streams,
                images, decode_states, decode_params, parse_time))
            return EXIT_FAILURE;

        std::cout << "." << std::flush;
        
        double start_decoding_time = wtime();
        CHECK_NVIMGCDCS(nvimgcdcsDecoderDecodeBatch(decoder, decode_state_batch,
            in_code_streams.data(), images.data(), params.batch_size, &decode_params, nullptr, true));
        double decode_time = wtime() - start_decoding_time;

        if (prepare_encode_resources(instance, current_names, encoder, encode_state_batch,
                is_host_input, is_device_input, out_code_streams, images, encode_states,
                encode_params, params, output_path))
            return EXIT_FAILURE;

        //CHECK_CUDA(cudaEventRecord(startEvent, NULL));
        double start_encoding_time = wtime();
        CHECK_NVIMGCDCS(nvimgcdcsEncoderEncodeBatch(encoder, encode_state_batch, images.data(),
            out_code_streams.data(), params.batch_size, &encode_params, nullptr, true));
        double encode_time = wtime() - start_encoding_time;
        //CHECK_CUDA(cudaEventRecord(stopEvent, NULL));
        //CHECK_CUDA(cudaEventSynchronize(stopEvent));
        //CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
       // double encode_time =
        //    0.001 * static_cast<double>(loopTime); // cudaEventElapsedTime returns milliseconds

        if (warmup < params.warmup) {
            warmup++;
        } else {
            total_processed += params.batch_size;
            total_reading_time += reading_time;
            total_parse_time += parse_time;
            total_decode_time += decode_time;
            total_encode_time += encode_time;
            total_processing_time += parse_time + decode_time + encode_time;
        }

        for (auto& cs : in_code_streams) {
            nvimgcdcsCodeStreamDestroy(cs);
        }
        for (auto& cs : out_code_streams) {
            nvimgcdcsCodeStreamDestroy(cs);
        }
    }

    for (int i = 0; i < params.batch_size; ++i) {
        nvimgcdcsImageDetachDecodeState(images[i]);
        nvimgcdcsImageDetachEncodeState(images[i]);
        nvimgcdcsImageDestroy(images[i]);
        nvimgcdcsDecodeStateDestroy(decode_states[i]);
        nvimgcdcsEncodeStateDestroy(encode_states[i]);
        if (image_buffers[i].data) {
            CHECK_CUDA(cudaFree(image_buffers[i].data));
        }
    }

    if (decoder) {
        nvimgcdcsDecoderDestroy(decoder);
    }

    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    total_time += wtime() - time;
    std::cout << std::endl;
    std::cout << "Total images: " << total_images << std::endl;
    std::cout << "Total transcoding time: " << total_time << std::endl;
    std::cout << "Avg transcoding time per image: " << total_time / total_images << std::endl;
    std::cout << "Avg transcode speed  (in images per sec): " << total_images / total_time
              << std::endl;
    std::cout << "Avg transcode time per batch: "
              << total_time / ((total_images + params.batch_size - 1) / params.batch_size)
              << std::endl;

    std::cout << "Total reading time: " << total_reading_time << std::endl;
    std::cout << "Avg reading time per image: " << total_reading_time / total_images << std::endl;
    std::cout << "Avg reading speed  (in images per sec): " << total_images / total_reading_time
              << std::endl;
    std::cout << "Avg reading time per batch: "
              << total_reading_time / ((total_images + params.batch_size - 1) / params.batch_size)
              << std::endl;

    std::cout << "Total parsing time: " << total_parse_time << std::endl;
    std::cout << "Avg parsing time per image: " << total_parse_time / total_images << std::endl;
    std::cout << "Avg parsing speed  (in images per sec): " << total_images / total_parse_time
              << std::endl;
    std::cout << "Avg parsing time per batch: "
              << total_parse_time / ((total_images + params.batch_size - 1) / params.batch_size)
              << std::endl;

    std::cout << "Total decoding time: " << total_decode_time << std::endl;
    std::cout << "Avg decoding time per image: " << total_decode_time / total_images << std::endl;
    std::cout << "Avg decoding speed  (in images per sec): " << total_images / total_decode_time
              << std::endl;
    std::cout << "Avg decoding time per batch: "
              << total_decode_time / ((total_images + params.batch_size - 1) / params.batch_size)
              << std::endl;

    std::cout << "Total encoding time: " << total_encode_time << std::endl;
    std::cout << "Avg encoding time per image: " << total_encode_time / total_images << std::endl;
    std::cout << "Avg encoding speed  (in images per sec): " << total_images / total_encode_time
              << std::endl;
    std::cout << "Avg encoding time per batch: "
              << total_encode_time / ((total_images + params.batch_size - 1) / params.batch_size)
              << std::endl;

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
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
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
    nvimgcdcsInstanceCreateInfo_t instance_create_info;
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
    nvimgcdcsExtension_t pxm_extension;
    nvimgcdcsExtensionDesc_t pxm_extension_desc;
    memset(&pxm_extension_desc, 0, sizeof(pxm_extension_desc));
    pxm_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;

    nvpxm::get_nvpxm_extension_desc(&pxm_extension_desc);
    nvimgcdcsExtensionCreate(instance, &pxm_extension, &pxm_extension_desc);

    fs::path exe_path(argv[0]);
    fs::path input_path = fs::absolute(exe_path).parent_path() / fs::path(params.input);
    fs::path output_path = fs::absolute(exe_path).parent_path() / fs::path(params.output);

    if (fs::is_directory(input_path)) {
        exit_code = process_images(instance, input_path, output_path, params);
    } else {
        exit_code = process_one_image(instance, input_path, output_path, params);
    }

    nvimgcdcsExtensionDestroy(pxm_extension);
    nvimgcdcsInstanceDestroy(instance);

    return exit_code;
}
