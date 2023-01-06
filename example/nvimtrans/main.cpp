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
#include "nvpxm.h"

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

double wtime(void)
{
#if defined(_WIN32)
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer        = QueryPerformanceFrequency(&t);
        oofreq                 = 1.0 / (double)t.QuadPart;
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

struct CommandLineParams
{
    std::string input;
    std::string output;
    std::string output_codec;
    int warmup;
    int verbose;
    float quality;
    float target_psnr;
    bool write_output;
    bool reversible;
    int num_decomps;
    int code_block_w;
    int code_block_h;
    bool dec_color_trans;
    bool enc_color_trans;
    bool optimized_huffman;
    bool ignore_orientation;
    nvimgcdcsJpegEncoding_t jpeg_encoding;
    nvimgcdcsChromaSubsampling_t chroma_subsampling;
};

int find_param_index(const char** argv, int argc, const char* parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++) {
        if (strncmp(argv[i], parm, 100) == 0) {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1) {
        return index;
    } else {
        std::cout << "Error, parameter " << parm << " has been specified more than once, exiting\n"
                  << std::endl;
        return -1;
    }

    return -1;
}

int process_commandline_params(int argc, const char* argv[], CommandLineParams* params)
{
    int pidx;
    if ((pidx = find_param_index(argv, argc, "-h")) != -1 ||
        (pidx = find_param_index(argv, argc, "--help")) != -1) {
        std::cout << "Usage: " << argv[0] << " [decoding options]"
                  << " -i <input> "
                  << "[encoding options]"
                  << " -o <output> " << std::endl;
        std::cout << std::endl;
        std::cout << "General options: " << std::endl;
        std::cout << "  -h\t\t: show help" << std::endl;
        std::cout << "  --help\t\t: show help" << std::endl;
        std::cout << "  -verbose\t\t: verbosity level from 0 to 5 (default 1)" << std::endl;
        std::cout << "  -w\t\t: warmup iterations (default 0)" << std::endl;
        std::cout << std::endl;
        std::cout << "Decoding options: " << std::endl;
        std::cout
            << "  -dec_color_trans\t: Decoding color transfrom. (default false)" << std::endl
            << "  \t\t\t - When true, for jpeg with 4 color components assumes CMYK colorspace "
               "and converts to RGB/YUV."
            << std::endl
            << "  \t\t\t - When true, for Jpeg2k and 422/420 chroma subsampling enable "
               "conversion to RGB."
            << std::endl;
        std::cout << "  -ignore_orientation\t: Ignore EXFIF orientation (default false)"
                  << std::endl;
        std::cout << "  -input\t\t: Path to single image" << std::endl;
        std::cout << std::endl;
        std::cout << "Encoding options: " << std::endl;
        std::cout << "  -output_codec\t\t: Output codec (default bmp)" << std::endl;
        std::cout << "  -quality\t\t: Quality to encode with (default 95)" << std::endl;
        std::cout << "  -chroma_subsampling\t: Chroma subsampling (default 444)" << std::endl;
        std::cout << "  -enc_color_trans\t: Encoding color transfrom. For true transform RGB "
                     "color images to YUV (default false)"
                  << std::endl;
        std::cout << "  -psnr\t\t\t: Target psnr (default 50)" << std::endl;
        std::cout << "  -reversible\t\t: false for lossy and true for lossless compresion (default "
                     "false) "
                  << std::endl;
        std::cout
            << "  -num_decomps\t\t: number of wavelet transform decompositions levels (default 5)"
            << std::endl;
        std::cout
            << "  -optimized_huffman\t: For false non-optimized Huffman will be used. Otherwise "
               "optimized version will be used. (default false)."
            << std::endl;
        std::cout << "  -jpeg_encoding\t: Corresponds to the JPEG marker"
                     " baseline_dct, sequential_dct or progressive_dct (default "
                     "baseline_dct)."
                  << std::endl;
        ;
        std::cout << "  -output\t\t: File to write decoded image using <output_codec>" << std::endl;

        return EXIT_SUCCESS;
    }
    params->warmup = 0;
    if ((pidx = find_param_index(argv, argc, "-w")) != -1) {
        params->warmup = static_cast<int>(strtod(argv[pidx + 1], NULL));
    }

    params->verbose = 1;
    if ((pidx = find_param_index(argv, argc, "-verbose")) != -1) {
        params->verbose = static_cast<int>(strtod(argv[pidx + 1], NULL));
    }

    params->input = "./";
    if ((pidx = find_param_index(argv, argc, "-i")) != -1) {
        params->input = argv[pidx + 1];
    } else {
        std::cout << "Please specify input directory with encoded images" << std::endl;
        return EXIT_FAILURE;
    }

    params->ignore_orientation = false;
    if ((pidx = find_param_index(argv, argc, "-ignore_orientation")) != -1) {
        params->ignore_orientation = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->quality = 95;
    if ((pidx = find_param_index(argv, argc, "-q")) != -1) {
        params->quality = static_cast<float>(strtod(argv[pidx + 1], NULL));
    }

    params->target_psnr = 50;
    if ((pidx = find_param_index(argv, argc, "-psnr")) != -1) {
        params->target_psnr = static_cast<float>(strtod(argv[pidx + 1], NULL));
    }

    params->output_codec = "bmp";
    if ((pidx = find_param_index(argv, argc, "-c")) != -1) {
        params->output_codec = argv[pidx + 1];
    }

    params->write_output = false;
    if ((pidx = find_param_index(argv, argc, "-o")) != -1) {
        params->output = argv[pidx + 1];
    }
    params->reversible = false;
    if ((pidx = find_param_index(argv, argc, "-reversible")) != -1) {
        params->reversible = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->num_decomps = 5;
    if ((pidx = find_param_index(argv, argc, "-num_decomps")) != -1) {
        params->num_decomps = atoi(argv[pidx + 1]);
    }

    params->code_block_w = 64;
    params->code_block_h = 64;
    if ((pidx = find_param_index(argv, argc, "-block_size")) != -1) {
        params->code_block_h = atoi(argv[pidx + 1]);
        params->code_block_w = atoi(argv[pidx + 2]);
    }
    params->dec_color_trans = false;
    if ((pidx = find_param_index(argv, argc, "-dec_color_trans")) != -1) {
        params->dec_color_trans = strcmp(argv[pidx + 1], "true") == 0;
    }
    params->enc_color_trans = false;
    if ((pidx = find_param_index(argv, argc, "-enc_color_trans")) != -1) {
        params->enc_color_trans = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->optimized_huffman = false;
    if ((pidx = find_param_index(argv, argc, "-optimized_huffman")) != -1) {
        params->optimized_huffman = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->jpeg_encoding = NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT;
    if ((pidx = find_param_index(argv, argc, "-jpeg_encoding")) != -1) {
        if (strcmp(argv[pidx + 1], "baseline_dct") == 0) {
            params->jpeg_encoding = NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT;
        } else if (strcmp(argv[pidx + 1], "sequential_dct") == 0) {
            params->jpeg_encoding = NVIMGCDCS_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN;
        } else if (strcmp(argv[pidx + 1], "progressive_dct") == 0) {
            params->jpeg_encoding = NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;
        } else {
            std::cout << "Unknown jpeg encoding type: " << argv[pidx + 1] << std::endl;
        }
    }
    params->chroma_subsampling = NVIMGCDCS_SAMPLING_444;
    if ((pidx = find_param_index(argv, argc, "-chroma_subsampling")) != -1) {
        std::map<std::string, nvimgcdcsChromaSubsampling_t> str2Css = {
            {"444", NVIMGCDCS_SAMPLING_444}, {"420", NVIMGCDCS_SAMPLING_420},
            {"440", NVIMGCDCS_SAMPLING_440}, {"422", NVIMGCDCS_SAMPLING_422},
            {"411", NVIMGCDCS_SAMPLING_411}, {"410", NVIMGCDCS_SAMPLING_410},
            {"gray", NVIMGCDCS_SAMPLING_GRAY}, {"410v", NVIMGCDCS_SAMPLING_410V}};
        auto it = str2Css.find(argv[pidx + 1]);
        if (it != str2Css.end()) {
            params->chroma_subsampling = it->second;
        } else {
            std::cout << "Unknown chroma subsampling type: " << argv[pidx + 1] << std::endl;
        }
    }
    return -1;
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

int main(int argc, const char* argv[])
{
    CommandLineParams params;
    int status = process_commandline_params(argc, argv, &params);
    if (status != -1) {
        return status;
    }

    double total_time  = 0.;
    double parse_time  = 0.;
    double decode_time = 0.;
    int total_images   = 1;

    namespace fs = std::filesystem;
    nvimgcdcsInstance_t instance;
    nvimgcdcsInstanceCreateInfo_t instance_create_info;
    instance_create_info.type                    = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.next                    = NULL;
    instance_create_info.pinned_allocator        = NULL;
    instance_create_info.device_allocator        = NULL;
    instance_create_info.load_extension_modules  = true;
    instance_create_info.default_debug_messenger = true;
    instance_create_info.message_severity        = verbosity2severity(params.verbose);
    instance_create_info.message_type            = NVIMGCDCS_DEBUG_MESSAGE_TYPE_ALL;

    nvimgcdcsInstanceCreate(&instance, instance_create_info);
    nvimgcdcsExtension_t pxm_extension;
    nvimgcdcsExtensionDesc_t pxm_extension_desc;
    memset(&pxm_extension_desc, 0, sizeof(pxm_extension_desc));
    pxm_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;

    nvpxm::get_nvpxm_extension_desc(&pxm_extension_desc);
    nvimgcdcsExtensionCreate(instance, &pxm_extension, &pxm_extension_desc);

    nvimgcdcsCodeStream_t code_stream;
    fs::path exe_path(argv[0]);
    fs::path input_file = fs::absolute(exe_path).parent_path() / fs::path(params.input);
    std::cout << "Loading " << input_file.string() << " file" << std::endl;
    std::ifstream file(input_file.string(), std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
    if (0) {
        nvimgcdcsCodeStreamCreateFromFile(instance, &code_stream, input_file.string().c_str());
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
    decode_params.enable_orientation      = !params.ignore_orientation;
    if (decode_params.enable_orientation) {
        decode_params.orientation.rotated = image_info.orientation.rotated == 90
                                                ? 270
                                                : (image_info.orientation.rotated == 270 ? 90 : 0);
        if (decode_params.orientation.rotated) {
            auto tmp                = image_info.image_width;
            image_info.image_width  = image_info.image_height;
            image_info.image_height = tmp;
        }
    }
    int bytes_per_element    = image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 1 : 2;
    image_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
    image_info.color_space   = NVIMGCDCS_COLORSPACE_SRGB;

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

    fs::path output_file = fs::absolute(exe_path).parent_path() / fs::path(params.output);
    std::cout << "Saving to " << output_file.string() << " file" << std::endl;

    nvimgcdcsCodeStream_t output_code_stream;
    nvimgcdcsCodeStreamCreateToFile(
        instance, &output_code_stream, output_file.string().c_str(), params.output_codec.data());
    nvimgcdcsCodeStreamSetImageInfo(output_code_stream, &image_info);

    image_info.sampling = params.chroma_subsampling;

    nvimgcdcsEncodeParams_t encode_params;
    memset(&encode_params, 0, sizeof(nvimgcdcsEncodeParams_t));
    encode_params.quality     = params.quality;
    encode_params.target_psnr = params.target_psnr;
    encode_params.mct_mode =
        params.enc_color_trans ? NVIMGCDCS_MCT_MODE_YCC : NVIMGCDCS_MCT_MODE_RGB;

    //codec sepcific encode params
    nvimgcdcsJpeg2kEncodeParams_t jpeg2k_encode_params;
    nvimgcdcsJpegEncodeParams_t jpeg_encode_params;
    if (params.output_codec == "jpeg2k") {
        memset(&jpeg2k_encode_params, 0, sizeof(jpeg2k_encode_params));
        jpeg2k_encode_params.type            = NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
        jpeg2k_encode_params.stream_type     = output_file.extension().string() == ".jp2"
                                                   ? NVIMGCDCS_JPEG2K_STREAM_JP2
                                                   : NVIMGCDCS_JPEG2K_STREAM_J2K;
        jpeg2k_encode_params.code_block_w    = params.code_block_w;
        jpeg2k_encode_params.code_block_h    = params.code_block_h;
        jpeg2k_encode_params.irreversible    = !params.reversible;
        jpeg2k_encode_params.prog_order      = NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL;
        jpeg2k_encode_params.num_resolutions = params.num_decomps;

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

        encode_params.next = &jpeg2k_encode_params;
    } else if (params.output_codec == "jpeg") {
        memset(&jpeg_encode_params, 0, sizeof(jpeg_encode_params));
        jpeg_encode_params.type              = NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS;
        jpeg_encode_params.encoding          = params.jpeg_encoding;
        jpeg_encode_params.optimized_huffman = params.optimized_huffman;
        encode_params.next                   = &jpeg_encode_params;
    }
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
        image_info.component_info[0].host_pitch_in_bytes =
            image_info.image_width * (is_interleaved ? image_info.num_components : 1);
        image_info.component_info[1].host_pitch_in_bytes =
            image_info.image_width * (is_interleaved ? image_info.num_components : 1);
        image_info.component_info[2].host_pitch_in_bytes =
            image_info.image_width * (is_interleaved ? image_info.num_components : 1);

        nvimgcdcsImageSetHostBuffer(image, host_buffer.data(), host_buffer.size());
    }
    if (is_device_output || is_device_input) {
        size_t device_pitch_in_bytes = 0;
        CHECK_CUDA(cudaMallocPitch((void**)&device_buffer, &device_pitch_in_bytes,
            image_info.image_width * bytes_per_element *
                (is_interleaved ? image_info.num_components : 1),
            image_info.image_height * (is_interleaved ? 1 : image_info.num_components)));
        image_info.component_info[0].device_pitch_in_bytes = device_pitch_in_bytes;
        image_info.component_info[1].device_pitch_in_bytes = device_pitch_in_bytes;
        image_info.component_info[2].device_pitch_in_bytes = device_pitch_in_bytes;
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
        nvimgcdcsDecoderDecode(decoder, code_stream, image, &decode_params);
        nvimgcdcsImage_t ready_decoded_image;
        nvimgcdcsProcessingStatus_t decode_status;
        nvimgcdcsInstanceGetReadyImage(instance, &ready_decoded_image, &decode_status, true);
        if (decode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
            std::cerr << "Error: Something went wrong during warmup decoding" << std::endl;
        }
        if (warmup_iter == (params.warmup - 1)) {
            std::cout << "Warmup done!" << std::endl;
        }
    }
    decode_time = wtime();
    nvimgcdcsDecoderDecode(decoder, code_stream, image, &decode_params);

    nvimgcdcsImage_t ready_decoded_image;
    nvimgcdcsProcessingStatus_t decode_status;
    nvimgcdcsInstanceGetReadyImage(instance, &ready_decoded_image, &decode_status, true);
    if (decode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong with decoding" << std::endl;
    }

    decode_time = wtime() - decode_time;

    assert(ready_decoded_image == image); //we sent only one image to decoder

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
    nvimgcdcsEncoderEncode(encoder, output_code_stream, image, &encode_params);

    nvimgcdcsImage_t ready_encoded_image;
    nvimgcdcsProcessingStatus_t encode_status;
    nvimgcdcsInstanceGetReadyImage(instance, &ready_encoded_image, &encode_status, true);
    if (decode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
        std::cerr << "Error: Something went wrong during encoding" << std::endl;
    }
    assert(ready_encoded_image == image);

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
    nvimgcdcsExtensionDestroy(pxm_extension);
    nvimgcdcsInstanceDestroy(instance);

    return EXIT_SUCCESS;
}