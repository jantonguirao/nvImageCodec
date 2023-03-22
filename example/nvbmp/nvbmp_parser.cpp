#include "nvbmp_parser.h"
#include <nvimgcodecs.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "log.h"

struct nvimgcdcsParser
{
    std::vector<nvimgcdcsCapability_t> capabilities_ = {NVIMGCDCS_CAPABILITY_HOST_INPUT};
};

static nvimgcdcsStatus_t nvbmp_parser_can_parse(
    void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    NVIMGCDCS_P_LOG_TRACE("nvbmp_parser_can_parse");

    constexpr size_t min_bmp_stream_size = 18u;
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    size_t length;
    io_stream->size(io_stream->instance, &length);
    if (length < min_bmp_stream_size) {
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    constexpr size_t signature_size = 2;
    std::vector<unsigned char> signature_buffer(signature_size);
    size_t output_size = 0;
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    io_stream->read(io_stream->instance, &output_size, &signature_buffer[0], signature_size);
    if (output_size != signature_size) {
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    *result = signature_buffer[0] == 'B' && signature_buffer[1] == 'M';
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_parser_create(
    void* instance, nvimgcdcsParser_t* parser)
{
    NVIMGCDCS_P_LOG_TRACE("nvbmp_parser_create");
    *parser = new nvimgcdcsParser();
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_parser_destroy(nvimgcdcsParser_t parser)
{
    NVIMGCDCS_P_LOG_TRACE("nvbmp_parser_destroy");
    delete parser;
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_create_parse_state(
    nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state)
{
    NVIMGCDCS_P_LOG_TRACE("nvbmp_create_parse_state");
    *parse_state = new nvimgcdcsParseState();
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_destroy_parse_state(nvimgcdcsParseState_t parse_state)
{
    NVIMGCDCS_P_LOG_TRACE("nvbmp_destroy_parse_state");
    delete parse_state;
    return NVIMGCDCS_STATUS_SUCCESS;
}

enum BmpCompressionType
{
    BMP_COMPRESSION_RGB = 0,
    BMP_COMPRESSION_RLE8 = 1,
    BMP_COMPRESSION_RLE4 = 2,
    BMP_COMPRESSION_BITFIELDS = 3
};

struct BitmapCoreHeader
{
    uint32_t header_size;
    uint16_t width, heigth, planes, bpp;
};
static_assert(sizeof(BitmapCoreHeader) == 12);

struct BitmapInfoHeader
{
    int32_t header_size;
    int32_t width, heigth;
    uint16_t planes, bpp;
    uint32_t compression, image_size;
    int32_t x_pixels_per_meter, y_pixels_per_meter;
    uint32_t colors_used, colors_important;
};
static_assert(sizeof(BitmapInfoHeader) == 40);

static bool is_color_palette(
    nvimgcdcsIoStreamDesc_t io_stream, size_t ncolors, int palette_entry_size)
{
    std::vector<uint8_t> entry;
    entry.resize(palette_entry_size);
    for (int i = 0; i < ncolors; i++) {
        size_t output_size;
        io_stream->read(io_stream->instance, &output_size, entry.data(), palette_entry_size);

        const auto b = entry[0], g = entry[1], r = entry[2]; // a = p[3];
        if (b != g || b != r)
            return true;
    }
    return false;
}

static int number_of_channels(nvimgcdcsIoStreamDesc_t io_stream, int bpp, int compression_type,
    size_t ncolors = 0, int palette_entry_size = 0)
{
    if (compression_type == BMP_COMPRESSION_RGB || compression_type == BMP_COMPRESSION_RLE8) {
        if (bpp <= 8 && ncolors <= static_cast<unsigned int>(1u << bpp)) {
            return is_color_palette(io_stream, ncolors, palette_entry_size) ? 3 : 1;
        } else if (bpp == 24) {
            return 3;
        } else if (bpp == 32) {
            return 4;
        }
    } else if (compression_type == BMP_COMPRESSION_BITFIELDS) {
        if (bpp == 16) {
            return 3;
        } else if (bpp == 32) {
            return 4;
        }
    }

    //throw std::runtime_error(make_string("configuration not supported. bpp: ", bpp,
    //    " compression_type:", compression_type, "ncolors:", ncolors));
    return 0;
}

static nvimgcdcsStatus_t nvbmp_parser_get_image_info(nvimgcdcsParser_t parser,
    nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    NVIMGCDCS_P_LOG_TRACE("nvbmp_parser_get_image_info");

    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    size_t length;
    io_stream->size(io_stream->instance, &length);
    if (length < 18u) {
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    static constexpr int kHeaderStart = 14;
    io_stream->seek(io_stream->instance, kHeaderStart, SEEK_SET);
    size_t output_size;

    io_stream->read(io_stream->instance, &output_size, &code_stream->parse_state->header_size,
        sizeof(code_stream->parse_state->header_size));
    io_stream->skip(io_stream->instance, -4);

    int bpp = 0;
    int compression_type = BMP_COMPRESSION_RGB;
    int ncolors = 0;
    int palette_entry_size = 0;
    size_t palette_start = 0;

    if (length >= 26 && code_stream->parse_state->header_size == 12) {
        BitmapCoreHeader header = {};
        io_stream->read(io_stream->instance, &output_size, &header, sizeof(header));
        image_info->width = header.width;
        image_info->height = header.heigth;
        bpp = header.bpp;
        if (bpp <= 8) {
            io_stream->tell(io_stream->instance, &palette_start);
            palette_entry_size = 3;
            ncolors = 1u << bpp;
        }
    } else if (length >= 50 && code_stream->parse_state->header_size >= 40) {
        BitmapInfoHeader header = {};
        io_stream->read(io_stream->instance, &output_size, &header, sizeof(header));
        io_stream->skip(
            io_stream->instance, code_stream->parse_state->header_size - sizeof(header));
        image_info->width = abs(header.width);
        image_info->height = abs(header.heigth);
        bpp = header.bpp;
        compression_type = header.compression;
        ncolors = header.colors_used;
        if (bpp <= 8) {
            io_stream->tell(io_stream->instance, &palette_start);
            palette_entry_size = 4;
            ncolors = ncolors == 0 ? 1u << bpp : ncolors;
        }
    } else {
        //const char* file_info = encoded->SourceInfo() ? encoded->SourceInfo() : "a file";
        //make_string("Unexpected length of a BMP header ", header_size, " in ", file_info,
        //    " which is ", length, " bytes long."));
    }

    // sanity check
    if (palette_start != 0) { // this silences a warning about unused variable
        assert(palette_start + (ncolors * palette_entry_size) <= length);
    }

    image_info->num_planes = number_of_channels(io_stream, bpp, compression_type, ncolors, palette_entry_size);
    for (size_t p = 0; p < image_info->num_planes; p++) {
        image_info->plane_info[p].height = image_info->height;
        image_info->plane_info[p].width = image_info->width;
        image_info->plane_info[p].num_channels = 1;
        image_info->plane_info[p].sample_type =
            NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8; // TODO Allow other sample data types
    }
    image_info->sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_get_capabilities(
    nvimgcdcsParser_t parser, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    NVIMGCDCS_P_LOG_TRACE("nvbmp_get_capabilities");
    if (parser == 0)
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;

    if (capabilities) {
        *capabilities = parser->capabilities_.data();
    }

    if (size) {
        *size = parser->capabilities_.size();
    } else {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsParserDesc nvbmp_parser = {
    NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC,
    NULL,
    NULL,               // instance    
    "nvbmp_parser",     // id
     0x00000100,        // version
    "bmp",              // codec_type 

    nvbmp_parser_can_parse,
    nvbmp_parser_create,
    nvbmp_parser_destroy,
    nvbmp_create_parse_state,
    nvbmp_destroy_parse_state,
    nvbmp_parser_get_image_info,
    nvbmp_get_capabilities
};
// clang-format on   