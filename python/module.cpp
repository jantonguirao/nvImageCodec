/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "module.h"
#include "image.h"

namespace nvimgcdcs {

Module::Module()
{
    nvimgcdcsInstanceCreateInfo_t instance_create_info{};
    instance_create_info.type = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.next = NULL;
    instance_create_info.pinned_allocator = NULL;
    instance_create_info.device_allocator = NULL;
    instance_create_info.load_builtin_modules = true;
    instance_create_info.load_extension_modules = true;
    instance_create_info.default_debug_messenger = false;
    instance_create_info.message_severity =
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL | NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR | NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING;
    instance_create_info.message_type = NVIMGCDCS_DEBUG_MESSAGE_TYPE_ALL;
    instance_create_info.num_cpu_threads = 10;
    nvimgcdcsInstanceCreate(&instance_, instance_create_info);
}

Module ::~Module()
{
    nvimgcdcsInstanceDestroy(instance_);
}

void Module::exportToPython(py::module& m, nvimgcdcsInstance_t instance)
{
    m.def(
        "imread",
        [instance](const char* file_name, const std::vector<int>& params) -> Image {
            std::vector<int> params_with_ending_zero = params;
            params_with_ending_zero.push_back(0);
            return Image(instance, file_name, params_with_ending_zero.data());
        },
        "Loads an image from a specified file", "file_name"_a, "params"_a = std::vector<int>());

    m.def(
        "imwrite",
        [instance](const char* file_name, const Image& image, const std::vector<int>& params) -> void {
            std::vector<int> params_with_ending_zero = params;
            params_with_ending_zero.push_back(0);
            nvimgcdcsImWrite(instance, image.getNvImgCdcsImage(), file_name, params_with_ending_zero.data());
        },
        "Saves an image to a specified file", "file_name"_a, "image"_a, "params"_a = std::vector<int>());

    m.def("asimage", [instance](py::handle src) -> Image { return Image(instance, src.ptr()); });

    m.attr("NVIMGCDCS_IMREAD_GRAYSCALE") = static_cast<int>(NVIMGCDCS_IMREAD_GRAYSCALE); // do not convert to RGB
    //for jpeg with 4 color components assumes CMYK colorspace and converts to RGB
    //for Jpeg2k and 422/420 chroma subsampling enable conversion to RGB
    m.attr("NVIMGCDCS_IMREAD_COLOR") = static_cast<int>(NVIMGCDCS_IMREAD_COLOR);
    m.attr("NVIMGCDCS_IMREAD_IGNORE_ORIENTATION") = static_cast<int>(NVIMGCDCS_IMREAD_IGNORE_ORIENTATION); //Ignore orientation from Exif;
    m.attr("NVIMGCDCS_IMREAD_DEVICE_ID") = static_cast<int>(NVIMGCDCS_IMREAD_DEVICE_ID);
    m.attr("NVIMGCDCS_IMWRITE_DEVICE_ID") = static_cast<int>(NVIMGCDCS_IMWRITE_DEVICE_ID);
    m.attr("NVIMGCDCS_IMWRITE_JPEG_QUALITY") = static_cast<int>(NVIMGCDCS_IMWRITE_JPEG_QUALITY);             // 0-100 default 95
    m.attr("NVIMGCDCS_IMWRITE_JPEG_PROGRESSIVE") = static_cast<int>(NVIMGCDCS_IMWRITE_JPEG_PROGRESSIVE);
    m.attr("NVIMGCDCS_IMWRITE_JPEG_OPTIMIZE") = static_cast<int>(NVIMGCDCS_IMWRITE_JPEG_OPTIMIZE);           //optimized_huffman
    m.attr("NVIMGCDCS_IMWRITE_JPEG_SAMPLING_FACTOR") = static_cast<int>(NVIMGCDCS_IMWRITE_JPEG_SAMPLING_FACTOR);
    m.attr("NVIMGCDCS_IMWRITE_JPEG2K_TARGET_PSNR") = static_cast<int>(NVIMGCDCS_IMWRITE_JPEG2K_TARGET_PSNR); // default 50
    m.attr("NVIMGCDCS_IMWRITE_JPEG2K_NUM_DECOMPS") = static_cast<int>(NVIMGCDCS_IMWRITE_JPEG2K_NUM_DECOMPS); // num_decomps default 5
    m.attr("NVIMGCDCS_IMWRITE_JPEG2K_CODE_BLOCK_SIZE") =
        static_cast<int>(NVIMGCDCS_IMWRITE_JPEG2K_CODE_BLOCK_SIZE); // code_block_w code_block_h (default 64 64)
    m.attr("NVIMGCDCS_IMWRITE_JPEG2K_REVERSIBLE") = static_cast<int>(NVIMGCDCS_IMWRITE_JPEG2K_REVERSIBLE);
    m.attr("NVIMGCDCS_IMWRITE_JPEG2K_PROG_ORDER") = static_cast<int>(NVIMGCDCS_IMWRITE_JPEG2K_PROG_ORDER);
    m.attr("NVIMGCDCS_JPEG2K_PROG_ORDER_LRCP") = static_cast<int>(NVIMGCDCS_JPEG2K_PROG_ORDER_LRCP);
    m.attr("NVIMGCDCS_JPEG2K_PROG_ORDER_RLCP") = static_cast<int>(NVIMGCDCS_JPEG2K_PROG_ORDER_RLCP);
    m.attr("NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL") = static_cast<int>(NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL);
    m.attr("NVIMGCDCS_JPEG2K_PROG_ORDER_PCRL") = static_cast<int>(NVIMGCDCS_JPEG2K_PROG_ORDER_PCRL);
    m.attr("NVIMGCDCS_JPEG2K_PROG_ORDER_CPRL") = static_cast<int>(NVIMGCDCS_JPEG2K_PROG_ORDER_CPRL);
    m.attr("NVIMGCDCS_IMWRITE_MCT_MODE") =
        static_cast<int>(NVIMGCDCS_IMWRITE_MCT_MODE); // nvimgcdcsMctMode_t value (default NVIMGCDCS_MCT_MODE_RGB )
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_444") = static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_444);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_422") = static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_422);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_420") = static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_420);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_440") = static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_440);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_411") = static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_411);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410") = static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410") = static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_410);
    m.attr("NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_GRAY") = static_cast<int>(NVIMGCDCS_IMWRITE_SAMPLING_FACTOR_GRAY);
}

} // namespace nvimgcdcs
