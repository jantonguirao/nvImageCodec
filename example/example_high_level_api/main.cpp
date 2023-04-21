#include <cuda_runtime_api.h>
#include <nvimgcodecs.h>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "command_line_params.h"

namespace fs = std::filesystem;

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

void get_read_flags(const CommandLineParams& params, std::vector<int>* read_params)
{
    if(params.ignore_orientation)
        read_params->push_back(NVIMGCDCS_IMREAD_IGNORE_ORIENTATION);
    if (params.dec_color_trans)
        read_params->push_back( NVIMGCDCS_IMREAD_COLOR);
    if (params.device_id != 0) {
        read_params->push_back(NVIMGCDCS_IMREAD_DEVICE_ID);
        read_params->push_back(params.device_id);
    }
}

void get_write_params(const CommandLineParams& params, std::vector<int>* write_params)
{
    if (params.output_codec == "jpeg") {
        write_params->push_back(NVIMGCDCS_IMWRITE_JPEG_QUALITY);
        write_params->push_back(static_cast<int>(params.quality));
        if (params.jpeg_encoding == NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)
            write_params->push_back(NVIMGCDCS_IMWRITE_JPEG_PROGRESSIVE);
        if (params.optimized_huffman)
            write_params->push_back(NVIMGCDCS_IMWRITE_JPEG_OPTIMIZE);
        write_params->push_back(NVIMGCDCS_IMWRITE_JPEG_SAMPLING_FACTOR);

        std::map<nvimgcdcsChromaSubsampling_t, nvimgcdcsImWriteSamplingFactor_t> css2sf = {
            {NVIMGCDCS_SAMPLING_444, NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_444},
            {NVIMGCDCS_SAMPLING_420, NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_420},
            {NVIMGCDCS_SAMPLING_440, NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_440},
            {NVIMGCDCS_SAMPLING_422, NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_422},
            {NVIMGCDCS_SAMPLING_411, NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_411},
            {NVIMGCDCS_SAMPLING_410, NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410},
            {NVIMGCDCS_SAMPLING_GRAY, NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_GRAY},
            {NVIMGCDCS_SAMPLING_410V, NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410V}};

        auto it = css2sf.find(params.chroma_subsampling);
        if (it != css2sf.end()) {
            write_params->push_back(it->second);
        } else {
            assert(!"MISSING CHROMA SUBSAMPLING VALUE");
            write_params->push_back(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_444);
        }

    } else if (params.output_codec == "jpeg2k") {
        write_params->push_back(NVIMGCDCS_IMWRITE_JPEG2K_TARGET_PSNR);
        assert(sizeof(float) == sizeof(int));
        int target_psnr;
        memcpy(&target_psnr, &params.target_psnr, sizeof(target_psnr));
        write_params->push_back(target_psnr);
        write_params->push_back(NVIMGCDCS_IMWRITE_JPEG2K_NUM_DECOMPS);
        write_params->push_back(params.num_decomps);
        write_params->push_back(NVIMGCDCS_IMWRITE_JPEG2K_CODE_BLOCK_SIZE);
        write_params->push_back(params.code_block_h);
        write_params->push_back(params.code_block_w);
        if (params.reversible)
            write_params->push_back(NVIMGCDCS_IMWRITE_JPEG2K_REVERSIBLE);
    }

    if (params.reversible)
        write_params->push_back(NVIMGCDCS_IMWRITE_JPEG2K_REVERSIBLE);

    if (params.device_id != 0) {
        write_params->push_back(NVIMGCDCS_IMWRITE_DEVICE_ID);
        write_params->push_back(params.device_id);
    }

    write_params->push_back(0);
}

void list_cuda_devices(int num_devices)
{
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
    CommandLineParams params;
    int status = process_commandline_params(argc, argv, &params);
    if (status != -1) {
        return status;
    }

    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (params.list_cuda_devices) {
        list_cuda_devices(num_devices);
    }

    if (params.device_id < num_devices) {
        cudaSetDevice(params.device_id);
    } else {
        std::cerr << "Error: Wrong device id #" << params.device_id << std::endl;
        list_cuda_devices(num_devices);
        return EXIT_FAILURE;
    }

    cudaDeviceProp props;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);
    std::cout << "Using GPU - " << props.name << " with Compute Capability " << props.major << "." << props.minor << std::endl;

    fs::path exe_path(argv[0]);
    fs::path input_file  = fs::absolute(exe_path).parent_path() / fs::path(params.input);
    fs::path output_file = fs::absolute(exe_path).parent_path() / fs::path(params.output);

    std::vector<int> read_params;
    get_read_flags(params, &read_params);
    std::vector<int> write_params;
    get_write_params(params, &write_params);

    nvimgcdcsInstance_t instance;
    nvimgcdcsInstanceCreateInfo_t instance_create_info{};
    instance_create_info.type             = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.next             = NULL;
    instance_create_info.pinned_allocator = NULL;
    instance_create_info.device_allocator = NULL;
    instance_create_info.load_builtin_modules    = true;
    instance_create_info.load_extension_modules  = true;
    instance_create_info.default_debug_messenger = true;
    instance_create_info.message_severity        = verbosity2severity(params.verbose);
    instance_create_info.message_type            = NVIMGCDCS_DEBUG_MESSAGE_TYPE_ALL;
    instance_create_info.num_cpu_threads = 10;
    nvimgcdcsInstanceCreate(&instance, instance_create_info);

    nvimgcdcsImage_t image;

    std::cout << "Loading " << input_file.string() << " file" << std::endl;
    nvimgcdcsImRead(instance, &image, input_file.string().c_str(), read_params.data());

    std::cout << "Saving to " << output_file.string() << " file" << std::endl;
    nvimgcdcsImWrite(instance, image, output_file.string().c_str(), write_params.data());

    nvimgcdcsImageDestroy(image);
    nvimgcdcsInstanceDestroy(instance);

    return EXIT_SUCCESS;
}