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

int main(int argc, const char* argv[])
{
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
    fs::path input_file = fs::absolute(exe_path).parent_path() / fs::path("input.j2k");
    std::cout << "Loading " << input_file.string() << " file" << std::endl;
    nvimgcdcsCodeStreamCreateFromFile(instance, &code_stream, input_file.string().c_str());
    nvimgcdcsImageInfo_t image_info;
    nvimgcdcsCodeStreamGetImageInfo(code_stream, &image_info);
    std::cout << "Image info: " << std::endl;
    std::cout << "\t - width:" << image_info.image_width << std::endl;
    std::cout << "\t - height:" << image_info.image_height << std::endl;
    std::cout << "\t - components:" << image_info.num_components << std::endl;
    int bytes_per_element  = image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 1 : 2;

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
    size_t image_buffer_size = image_info.component_info[0].pitch_in_bytes *
                               image_info.image_height * image_info.num_components;

    nvimgcdcsImage_t image;
    nvimgcdcsImageCreate(instance, &image, &image_info);
    nvimgcdcsImageSetDeviceBuffer(image, image_buffer, image_buffer_size);
    nvimgcdcsImageAttachDecodeState(image, decode_state);

    nvimgcdcsDecoderDecode(decoder, code_stream, image, &decode_params);
    cudaDeviceSynchronize();

    constexpr std::string_view output_codec("bmp");
    nvimgcdcsCodeStream_t bmp_code_stream;
    fs::path output_file = fs::absolute(exe_path).parent_path() / fs::path("output.bmp");
    std::cout << "Saving to " << output_file.string() << " file" << std::endl;
    nvimgcdcsCodeStreamCreateToFile(
        instance, &bmp_code_stream, output_file.string().c_str(), output_codec.data());
    nvimgcdcsCodeStreamSetImageInfo(bmp_code_stream, &image_info);

    nvimgcdcsEncodeParams_t encode_params;
    encode_params.backend.useCPU = true;
    encode_params.codec          = output_codec.data();

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