#include <cuda_runtime_api.h>
#include <nvimgcodecs.h>
#include <filesystem>
#include <iostream>
#include <cstring>
namespace fs = std::filesystem;


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

    fs::path exe_path(argv[0]);
    fs::path input_file = fs::absolute(exe_path).parent_path() / fs::path(params.input);
    fs::path output_file = fs::absolute(exe_path).parent_path() / fs::path(params.output);

    nvimgcdcsInstance_t instance;
    nvimgcdcsInstanceCreateInfo_t instance_create_info;
    instance_create_info.type             = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.next             = NULL;
    instance_create_info.pinned_allocator = NULL;
    instance_create_info.device_allocator = NULL;

    nvimgcdcsInstanceCreate(&instance, instance_create_info);
    
    nvimgcdcsImage_t image;
    std::cout << "Loading " << input_file.string() << " file" << std::endl;
    nvimgcdcsImgRead(instance, &image, input_file.string().c_str());

    std::cout << "Saving to " << output_file.string() << " file" << std::endl;
    nvimgcdcsImgWrite(instance, image, output_file.string().c_str(), NULL);

    nvimgcdcsImageDestroy(image);
    nvimgcdcsInstanceDestroy(instance);

    return EXIT_SUCCESS;
}