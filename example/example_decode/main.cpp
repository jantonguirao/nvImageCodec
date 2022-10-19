#include <nvimgcodecs.h>
#include <filesystem>
#include <iostream>

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

    nvimgcdcsDecodeParams_t decode_params;
    decode_params.backend.useGPU = true;
   
    nvimgcdcsDecoder_t decoder;
    nvimgcdcsDecoderCreate(instance, &decoder, code_stream, &decode_params);

    nvimgcdcsDecodeState_t decode_state;
    nvimgcdcsDecodeStateCreate(decoder, &decode_state);

    nvimgcdcsDecodeStateDestroy(decode_state);
    nvimgcdcsDecoderDestroy(decoder);
    nvimgcdcsCodeStreamDestroy(code_stream);
    nvimgcdcsInstanceDestroy(instance);

    return 0;
}