#include <cuda_runtime_api.h>
#include <nvimgcodecs.h>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct CommandLineParams
{
    std::string input;
    std::string output;
    std::string output_codec;
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
        std::cout << "  -h\t\t: show help" << std::endl;
        std::cout << "  --help\t\t: show help" << std::endl;
        std::cout << "  -verbose\t\t: verbosity level from 0 to 5 (default 1)" << std::endl;
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


    params->write_output = false;
    if ((pidx = find_param_index(argv, argc, "-o")) != -1) {
        params->output = argv[pidx + 1];
    }

    params->output_codec = "bmp";
    std::filesystem::path file_path(params->output);
    if (file_path.has_extension()) {
        std::string extension = file_path.extension().string();
        auto it               = ext2codec.find(extension);
        if (it != ext2codec.end()) {
            params->output_codec = it->second;
        }
    }
    if ((pidx = find_param_index(argv, argc, "-c")) != -1) {
        params->output_codec = argv[pidx + 1];
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

int get_read_flags(const CommandLineParams& params)
{
    int flags = 0;
    flags |= params.ignore_orientation ? NVIMGCDCS_IMREAD_IGNORE_ORIENTATION : 0;
    flags |= params.dec_color_trans ? NVIMGCDCS_IMREAD_COLOR : 0;
    return flags;
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

        std::map<nvimgcdcsChromaSubsampling_t, nvimgcdcsImwriteSamplingFactor_t> css2sf = {
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

    write_params->push_back(0);
}

int main(int argc, const char* argv[])
{
    CommandLineParams params;
    int status = process_commandline_params(argc, argv, &params);
    if (status != -1) {
        return status;
    }

    fs::path exe_path(argv[0]);
    fs::path input_file  = fs::absolute(exe_path).parent_path() / fs::path(params.input);
    fs::path output_file = fs::absolute(exe_path).parent_path() / fs::path(params.output);

    int read_flags = get_read_flags(params);
    std::vector<int> write_params;
    get_write_params(params, &write_params);

    nvimgcdcsInstance_t instance;
    nvimgcdcsInstanceCreateInfo_t instance_create_info;
    instance_create_info.type             = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.next             = NULL;
    instance_create_info.pinned_allocator = NULL;
    instance_create_info.device_allocator = NULL;
    instance_create_info.load_extension_modules  = true;
    instance_create_info.default_debug_messenger = true;
    instance_create_info.message_severity        = verbosity2severity(params.verbose);
    instance_create_info.message_type            = NVIMGCDCS_DEBUG_MESSAGE_TYPE_ALL;
    nvimgcdcsInstanceCreate(&instance, instance_create_info);

    nvimgcdcsImage_t image;

    std::cout << "Loading " << input_file.string() << " file" << std::endl;
    nvimgcdcsImRead(instance, &image, input_file.string().c_str(), read_flags);

    std::cout << "Saving to " << output_file.string() << " file" << std::endl;
    nvimgcdcsImWrite(instance, image, output_file.string().c_str(), write_params.data());

    nvimgcdcsImageDestroy(image);
    nvimgcdcsInstanceDestroy(instance);

    return EXIT_SUCCESS;
}