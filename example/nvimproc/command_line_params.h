/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodecs.h>
#include <filesystem>
#include <string>
#include <map>

namespace fs = std::filesystem;

struct CommandLineParams
{
    std::string input;
    std::string output;
    std::string output_codec;
    int warmup;
    int batch_size;
    int total_images;
    int device_id;
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
    bool list_cuda_devices;
};

#ifdef COMMAND_PARAMS_IMPLEMENTATION
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
    static std::map<std::string, std::string> ext2codec = {{".bmp", "bmp"}, {".j2c", "jpeg2k"},
        {".j2k", "jpeg2k"}, {".jp2", "jpeg2k"}, {".tiff", "tiff"}, {".tif", "tiff"},
        {".jpg", "jpeg"}, {".jpeg", "jpeg"}, {".ppm", "pxm"}, {".pgm", "pxm"}, {".pbm", "pxm"}};

    int pidx;
    if ((pidx = find_param_index(argv, argc, "-h")) != -1 ||
        (pidx = find_param_index(argv, argc, "--help")) != -1) {
        std::cout << "Usage: " << argv[0] << " [decoding options]"
                  << " -i <input> "
                  << "[encoding options]"
                  << " -o <output> " << std::endl;
        std::cout << std::endl;
        std::cout << "General options: " << std::endl;
        std::cout << "  -h --help\t\t: show help" << std::endl;
        std::cout << "  -v --verbose\t\t: verbosity level from 0 to 5 (default 1)" << std::endl;
        std::cout << "  -w\t\t\t: warmup iterations (default 0)" << std::endl;
        std::cout << "  -l\t\t\t: List cuda devices" << std::endl;
        std::cout << "  -b --batch_size\t: Batch size (default 1)" << std::endl;
        std::cout << std::endl;
        std::cout << "Decoding options: " << std::endl;
        std::cout
            << "  --dec_color_trans\t: Decoding color transfrom. (default false)" << std::endl
            << "  \t\t\t - When true, for jpeg with 4 color components assumes CMYK colorspace "
               "and converts to RGB/YUV."
            << std::endl
            << "  \t\t\t - When true, for Jpeg2k and 422/420 chroma subsampling enable "
               "conversion to RGB."
            << std::endl;
        std::cout << "  --ignore_orientation\t: Ignore EXFIF orientation (default false)"
                  << std::endl;
        std::cout << "  -i  --input\t\t: Path to single image or directory" << std::endl;
        std::cout << std::endl;
        std::cout << "Encoding options: " << std::endl;
        std::cout << "  -c --output_codec\t: Output codec (default bmp)" << std::endl;
        std::cout << "  -q --quality\t\t: Quality to encode with (default 95)" << std::endl;
        std::cout << "  --chroma_subsampling\t: Chroma subsampling (default 444)" << std::endl;
        std::cout << "  --enc_color_trans\t: Encoding color transfrom. For true transform RGB "
                     "color images to YUV (default false)"
                  << std::endl;
        std::cout << "  --psnr\t\t: Target psnr (default 50)" << std::endl;
        std::cout
            << "  --reversible\t\t: false for lossy and true for lossless compresion (default "
               "false) "
            << std::endl;
        std::cout
            << "  --num_decomps\t\t: number of wavelet transform decompositions levels (default 5)"
            << std::endl;
        std::cout
            << "  --optimized_huffman\t: For false non-optimized Huffman will be used. Otherwise "
               "optimized version will be used. (default false)."
            << std::endl;
        std::cout << "  --jpeg_encoding\t: Corresponds to the JPEG marker"
                     " baseline_dct, sequential_dct or progressive_dct (default "
                     "baseline_dct)."
                  << std::endl;
        ;
        std::cout
            << "  -o  --output\t\t: File or directory to write decoded image using <output_codec>"
            << std::endl;

        return EXIT_SUCCESS;
    }
    params->warmup = 0;
    if ((pidx = find_param_index(argv, argc, "-w")) != -1) {
        params->warmup = static_cast<int>(strtod(argv[pidx + 1], NULL));
    }

    params->verbose = 1;
    if (((pidx = find_param_index(argv, argc, "--verbose")) != -1) ||
        ((pidx = find_param_index(argv, argc, "-v")) != -1)) {
        params->verbose = static_cast<int>(strtod(argv[pidx + 1], NULL));
    }

    params->input = "./";
    if ((pidx = find_param_index(argv, argc, "-i")) != -1 ||
        (pidx = find_param_index(argv, argc, "--input")) != -1) {
        params->input = argv[pidx + 1];
    } else {
        std::cout << "Please specify input directory with encoded images" << std::endl;
        return EXIT_FAILURE;
    }

    params->ignore_orientation = false;
    if ((pidx = find_param_index(argv, argc, "--ignore_orientation")) != -1) {
        params->ignore_orientation = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->quality = 95;
    if ((pidx = find_param_index(argv, argc, "-q")) != -1 ||
        (pidx = find_param_index(argv, argc, "--quality")) != -1) {
        params->quality = static_cast<float>(strtod(argv[pidx + 1], NULL));
    }

    params->target_psnr = 50;
    if ((pidx = find_param_index(argv, argc, "--psnr")) != -1) {
        params->target_psnr = static_cast<float>(strtod(argv[pidx + 1], NULL));
    }

    params->write_output = false;
    if ((pidx = find_param_index(argv, argc, "-o")) != -1 ||
        (pidx = find_param_index(argv, argc, "--output")) != -1) {
        params->output = argv[pidx + 1];
    }

    params->output_codec = "bmp";
    fs::path file_path(params->output);
    if (file_path.has_extension()) {
        std::string extension = file_path.extension().string();
        auto it               = ext2codec.find(extension);
        if (it != ext2codec.end()) {
            params->output_codec = it->second;
        }
    }
    if ((pidx = find_param_index(argv, argc, "-c")) != -1 ||
        (pidx = find_param_index(argv, argc, "--output_codec")) != -1) {
        params->output_codec = argv[pidx + 1];
    }

    params->reversible = false;
    if ((pidx = find_param_index(argv, argc, "--reversible")) != -1) {
        params->reversible = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->num_decomps = 5;
    if ((pidx = find_param_index(argv, argc, "--num_decomps")) != -1) {
        params->num_decomps = atoi(argv[pidx + 1]);
    }

    params->code_block_w = 64;
    params->code_block_h = 64;
    if ((pidx = find_param_index(argv, argc, "--block_size")) != -1) {
        params->code_block_h = atoi(argv[pidx + 1]);
        params->code_block_w = atoi(argv[pidx + 2]);
    }
    params->dec_color_trans = false;
    if ((pidx = find_param_index(argv, argc, "--dec_color_trans")) != -1) {
        params->dec_color_trans = strcmp(argv[pidx + 1], "true") == 0;
    }
    params->enc_color_trans = false;
    if ((pidx = find_param_index(argv, argc, "--enc_color_trans")) != -1) {
        params->enc_color_trans = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->optimized_huffman = false;
    if ((pidx = find_param_index(argv, argc, "--optimized_huffman")) != -1) {
        params->optimized_huffman = strcmp(argv[pidx + 1], "true") == 0;
    }

    params->jpeg_encoding = NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT;
    if ((pidx = find_param_index(argv, argc, "--jpeg_encoding")) != -1) {
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
    if ((pidx = find_param_index(argv, argc, "--chroma_subsampling")) != -1) {
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

    params->batch_size = 1;
    if ((pidx = find_param_index(argv, argc, "-b")) != -1 ||
        (pidx = find_param_index(argv, argc, "--batch_size")) != -1) {
        params->batch_size = std::atoi(argv[pidx + 1]);
    }

    params->total_images = -1;
    if ((pidx = find_param_index(argv, argc, "-t")) != -1) {
        params->total_images = std::atoi(argv[pidx + 1]);
    }

    params->list_cuda_devices = find_param_index(argv, argc, "-l") != -1;

    return -1;
}
#endif