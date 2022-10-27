#include <cuda_runtime_api.h>
#include <nvimgcodecs.h>
#include <filesystem>
#include <iostream>
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
                  << "[-c output_codec] "
                  << std::endl;

        std::cout << "Parameters: " << std::endl;
        std::cout << "\timages_dir\t:\tPath to single image or directory of images" << std::endl;
        std::cout << "\toutput_dir\t:\tWrite decoded images using <output_codec> to this directory" << std::endl;
        std::cout << "\toutput_codec (defualt:bmp)\t: Output codec"<< std::endl;

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

    image_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;

    nvimgcdcsDecodeParams_t decode_params;
    decode_params.backend.useGPU = true;

    nvimgcdcsDecoder_t decoder;
    nvimgcdcsDecoderCreate(instance, &decoder, code_stream, &decode_params);

    nvimgcdcsDecodeState_t decode_state;
    nvimgcdcsDecodeStateCreate(decoder, &decode_state);

    unsigned char* image_buffer;
    CHECK_CUDA(cudaMallocPitch((void**)&image_buffer, &image_info.component_info[0].pitch_in_bytes,
        image_info.image_width * bytes_per_element,
        image_info.image_height * image_info.num_components));
    image_info.component_info[1].pitch_in_bytes = image_info.component_info[0].pitch_in_bytes;
    image_info.component_info[2].pitch_in_bytes = image_info.component_info[0].pitch_in_bytes;
    size_t image_buffer_size                    = image_info.component_info[0].pitch_in_bytes *
                               image_info.image_height * image_info.num_components;
    nvimgcdcsImage_t image;
    nvimgcdcsImageCreate(instance, &image, &image_info);
    nvimgcdcsImageSetDeviceBuffer(image, image_buffer, image_buffer_size);

    std::vector<unsigned char> host_buffer;
    if (std::string_view(codec_name) == "bmp") {
        host_buffer.resize(
            image_info.image_width * image_info.image_height * image_info.num_components);
        nvimgcdcsImageSetHostBuffer(image, host_buffer.data(), host_buffer.size());
    }

    nvimgcdcsImageAttachDecodeState(image, decode_state);

    nvimgcdcsDecoderDecode(decoder, code_stream, image, &decode_params);
    cudaDeviceSynchronize();

    nvimgcdcsCodeStream_t bmp_code_stream;
    fs::path output_file = fs::absolute(exe_path).parent_path() / fs::path(params.output);
    std::cout << "Saving to " << output_file.string() << " file" << std::endl;
    nvimgcdcsCodeStreamCreateToFile(instance, &bmp_code_stream, output_file.string().c_str(),
        params.output_codec.data());
    nvimgcdcsCodeStreamSetImageInfo(bmp_code_stream, &image_info);

    nvimgcdcsEncodeParams_t encode_params;
    encode_params.backend.useCPU = true;
    encode_params.target_psnr    = 50;
    encode_params.codec          = params.output_codec.data();

    nvimgcdcsEncoder_t encoder;
    nvimgcdcsEncoderCreate(instance, &encoder, bmp_code_stream, &encode_params);
    nvimgcdcsEncodeState_t encode_state;
    nvimgcdcsEncodeStateCreate(encoder, &encode_state);
    nvimgcdcsImageAttachEncodeState(image, encode_state);
    nvimgcdcsEncoderEncode(encoder, bmp_code_stream, image, &encode_params);

    nvimgcdcsImageDetachEncodeState(image);
    nvimgcdcsEncodeStateDestroy(encode_state);
    nvimgcdcsEncoderDestroy(encoder);
    nvimgcdcsCodeStreamDestroy(bmp_code_stream);

    nvimgcdcsImageDetachDecodeState(image);
    CHECK_CUDA(cudaFree(image_buffer));
    nvimgcdcsImageDestroy(image);
    nvimgcdcsDecodeStateDestroy(decode_state);
    nvimgcdcsDecoderDestroy(decoder);
    nvimgcdcsCodeStreamDestroy(code_stream);
    nvimgcdcsInstanceDestroy(instance);

    return EXIT_SUCCESS;
}