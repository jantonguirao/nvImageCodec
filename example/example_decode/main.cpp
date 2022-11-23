#include <cuda_runtime_api.h>
#include <nvimgcodecs.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
//#include <span>
#include <vector>

#define CHECK_CUDA(call)                                                                \
    {                                                                                   \
        cudaError_t _e = (call);                                                        \
        if (_e != cudaSuccess) {                                                        \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                         \
            return EXIT_FAILURE;                                                        \
        }                                                                               \
    }

struct CommandLineParams
{
    std::string input;
    std::string output;
    std::string output_codec;
    bool write_output;
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
        std::cout << "Usage: " << argv[0] << " -i images_dir "
                  << "[-o output_dir] "
                  << "[-c output_codec] " << std::endl;

        std::cout << "Parameters: " << std::endl;
        std::cout << "\timages_dir\t:\tPath to single image or directory of images" << std::endl;
        std::cout << "\toutput_dir\t:\tWrite decoded images using <output_codec> to this directory"
                  << std::endl;
        std::cout << "\toutput_codec (defualt:bmp)\t: Output codec" << std::endl;

        return EXIT_SUCCESS;
    }
    params->input = "./";
    if ((pidx = find_param_index(argv, argc, "-i")) != -1) {
        params->input = argv[pidx + 1];
    } else {
        std::cout << "Please specify input directory with encoded images" << std::endl;
        return EXIT_FAILURE;
    }
    params->write_output = false;
    if ((pidx = find_param_index(argv, argc, "-o")) != -1) {
        params->output = argv[pidx + 1];
    }
    params->write_output = true;
    params->output_codec = "bmp";
    if ((pidx = find_param_index(argv, argc, "-c")) != -1) {
        params->output_codec = argv[pidx + 1];
    }
    return -1;
}

int main(int argc, const char* argv[])
{
    CommandLineParams params;
    int status = process_commandline_params(argc, argv, &params);
    if (status != -1) {
        return status;
    }

    namespace fs = std::filesystem;
    nvimgcdcsInstance_t instance;
    nvimgcdcsInstanceCreateInfo_t instance_create_info;
    instance_create_info.type             = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.next             = NULL;
    instance_create_info.pinned_allocator = NULL;
    instance_create_info.device_allocator = NULL;

    nvimgcdcsInstanceCreate(&instance, instance_create_info);
    nvimgcdcsCodeStream_t code_stream;
    fs::path exe_path(argv[0]);
    fs::path input_file = fs::absolute(exe_path).parent_path() / fs::path(params.input);
    std::cout << "Loading " << input_file.string() << " file" << std::endl;
    nvimgcdcsCodeStreamCreateFromFile(instance, &code_stream, input_file.string().c_str());
    nvimgcdcsImageInfo_t image_info;
    nvimgcdcsCodeStreamGetImageInfo(code_stream, &image_info);
    char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
    nvimgcdcsCodeStreamGetCodecName(code_stream, codec_name);
    std::cout << "Input image info: " << std::endl;
    std::cout << "\t - width:" << image_info.image_width << std::endl;
    std::cout << "\t - height:" << image_info.image_height << std::endl;
    std::cout << "\t - components:" << image_info.num_components << std::endl;
    std::cout << "\t - codec:" << codec_name << std::endl;
    int bytes_per_element = image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 1 : 2;
    nvimgcdcsDecodeParams_t decode_params;
    memset(&decode_params, 0, sizeof(nvimgcdcsDecodeParams_t));

    nvimgcdcsDecoder_t decoder;
    nvimgcdcsDecoderCreate(instance, &decoder, code_stream, &decode_params);

    nvimgcdcsDecodeState_t decode_state;
    nvimgcdcsDecodeStateCreate(decoder, &decode_state, nullptr);

    nvimgcdcsImage_t image;
    nvimgcdcsImageCreate(instance, &image);

    unsigned char* device_buffer = nullptr;
    std::vector<unsigned char> host_buffer;

    size_t capabilities_size;
    nvimgcdcsDecoderGetCapabilities(decoder, nullptr, &capabilities_size);
    const nvimgcdcsCapability_t* capabilities_ptr;
    nvimgcdcsDecoderGetCapabilities(decoder, &capabilities_ptr, &capabilities_size);
#if 0    
    std::span<const nvimgcdcsCapability_t> decoder_capabilties{capabilities_ptr, capabilities_size};


    bool is_host_output   = std::find(decoder_capabilties.begin(), decoder_capabilties.end(),
                                NVIMGCDCS_CAPABILITY_HOST_OUTPUT) != decoder_capabilties.end();
    bool is_device_output = std::find(decoder_capabilties.begin(), decoder_capabilties.end(),
                                NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT) != decoder_capabilties.end();
#else
    bool is_host_output = std::find(capabilities_ptr,
                              capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                              NVIMGCDCS_CAPABILITY_HOST_OUTPUT) !=
                          capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
    bool is_device_output =
        std::find(capabilities_ptr,
            capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
            NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT) !=
        capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
#endif
    nvimgcdcsCodeStream_t output_code_stream;
    fs::path output_file = fs::absolute(exe_path).parent_path() / fs::path(params.output);
    std::cout << "Saving to " << output_file.string() << " file" << std::endl;
    nvimgcdcsCodeStreamCreateToFile(
        instance, &output_code_stream, output_file.string().c_str(), params.output_codec.data());
    nvimgcdcsCodeStreamSetImageInfo(output_code_stream, &image_info);

    nvimgcdcsEncodeParams_t encode_params;
    memset(&encode_params, 0, sizeof(nvimgcdcsEncodeParams_t));
    //TODO define and pass params
    encode_params.qstep       = 75;
    encode_params.target_psnr = 50;
    encode_params.codec       = params.output_codec.data();

    nvimgcdcsEncoder_t encoder = nullptr;
    nvimgcdcsEncoderCreate(instance, &encoder, output_code_stream, &encode_params);

    nvimgcdcsEncoderGetCapabilities(encoder, nullptr, &capabilities_size);
    nvimgcdcsEncoderGetCapabilities(encoder, &capabilities_ptr, &capabilities_size);
#if 0
    std::span<const nvimgcdcsCapability_t> encoder_capabilties{capabilities_ptr, capabilities_size};

    bool is_host_input   = std::find(encoder_capabilties.begin(), encoder_capabilties.end(),
                               NVIMGCDCS_CAPABILITY_HOST_INPUT) != encoder_capabilties.end();
    bool is_device_input = std::find(encoder_capabilties.begin(), encoder_capabilties.end(),
                               NVIMGCDCS_CAPABILITY_DEVICE_INPUT) != encoder_capabilties.end();
#else
    bool is_host_input = std::find(capabilities_ptr,
                             capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                             NVIMGCDCS_CAPABILITY_HOST_INPUT) !=
                         capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
    bool is_device_input = std::find(capabilities_ptr,
                               capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t),
                               NVIMGCDCS_CAPABILITY_DEVICE_INPUT) !=
                           capabilities_ptr + capabilities_size * sizeof(nvimgcdcsCapability_t);
#endif
    bool is_interleaved = static_cast<int>(image_info.sample_format) % 2 == 0;

    if (is_host_output || is_host_input) {
        host_buffer.resize(image_info.image_width * image_info.image_height *
                           image_info.num_components); //TODO more bytes per sample
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

    nvimgcdcsDecoderDecode(decoder, code_stream, image, &decode_params);

    nvimgcdcsImage_t ready_decoded_image;
    nvimgcdcsProcessingStatus_t decode_status;
    nvimgcdcsInstanceGetReadyImage(instance, &ready_decoded_image, &decode_status, true);
    if (decode_status != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
        std::cout << "Error:Something went wrong with decoding" << std::endl;
    }

    assert(ready_decoded_image == image); //we sent only one image to decoder

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
        std::cout << "Error:Something went wrong with encoding" << std::endl;
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
    nvimgcdcsInstanceDestroy(instance);

    return EXIT_SUCCESS;
}