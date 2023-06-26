/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "parsers/bmp.h"
#include <nvimgcodecs.h>
#include <vector>
#include <string.h>

#include "exception.h"
#include "log.h"
#include "logger.h"
#include "parsers/byte_io.h"
#include "parsers/exif.h"

namespace nvimgcdcs {

namespace {

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

static bool is_color_palette(nvimgcdcsIoStreamDesc_t io_stream, size_t ncolors, int palette_entry_size)
{
    std::vector<uint8_t> entry;
    entry.resize(palette_entry_size);
    for (size_t i = 0; i < ncolors; i++) {
        size_t output_size;
        io_stream->read(io_stream->instance, &output_size, entry.data(), palette_entry_size);

        const auto b = entry[0], g = entry[1], r = entry[2]; // a = p[3];
        if (b != g || b != r)
            return true;
    }
    return false;
}

static int number_of_channels(
    nvimgcdcsIoStreamDesc_t io_stream, int bpp, int compression_type, size_t ncolors = 0, int palette_entry_size = 0)
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
    return 0;
}

} // namespace

BMPParserPlugin::BMPParserPlugin()
    : parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, nullptr,
          this,         // instance
          "bmp_parser", // id
          "bmp",        // codec_type
          static_can_parse, static_create, Parser::static_destroy, Parser::static_get_image_info}
{
}

struct nvimgcdcsParserDesc* BMPParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t BMPParserPlugin::canParse(bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    constexpr size_t min_bmp_stream_size = 18u;
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    size_t length;
    io_stream->size(io_stream->instance, &length);
    if (length < min_bmp_stream_size) {
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    std::array<uint8_t, 2> signature;
    size_t output_size = 0;
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    io_stream->read(io_stream->instance, &output_size, signature.data(), signature.size());
    if (output_size != signature.size()) {
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    *result = signature[0] == 'B' && signature[1] == 'M';
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t BMPParserPlugin::static_can_parse(void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("bmp_parser_can_parse");
        CHECK_NULL(instance);
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
        auto handle = reinterpret_cast<BMPParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not check if code stream can be parsed - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

BMPParserPlugin::Parser::Parser()
{
}

nvimgcdcsStatus_t BMPParserPlugin::create(nvimgcdcsParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcdcsParser_t>(new BMPParserPlugin::Parser());
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t BMPParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("bmp_parser_create");
        CHECK_NULL(instance);
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<BMPParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not create bmp parser - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t BMPParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("bmp_parser_destroy");
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<BMPParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not destroy bmp parser - " << e.what());
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t BMPParserPlugin::Parser::getImageInfo(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    // https://en.wikipedia.org/wiki/BMP_file_format#DIB_header_(bitmap_information_header)
    NVIMGCDCS_LOG_TRACE("bmp_parser_get_image_info");
    try {
        nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
        size_t length;
        io_stream->size(io_stream->instance, &length);
        if (length < 18u) {
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }

        if (image_info->type != NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO) {
            NVIMGCDCS_LOG_ERROR("Unexpected structure type");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        strcpy(image_info->codec_name, "bmp");
        constexpr size_t header_start = 14;
        io_stream->seek(io_stream->instance, header_start, SEEK_SET);
        uint32_t header_size = ReadValueLE<uint32_t>(io_stream);
        // we'll read it again - it's part of the header struct
        io_stream->seek(io_stream->instance, header_start, SEEK_SET);

        int bpp = 0;
        int compression_type = BMP_COMPRESSION_RGB;
        int ncolors = 0;
        int palette_entry_size = 0;
        size_t palette_start = 0;

        if (length >= 26 && header_size == 12) {
            BitmapCoreHeader header = ReadValue<BitmapCoreHeader>(io_stream);
            image_info->plane_info[0].width = header.width;
            image_info->plane_info[0].height = header.heigth;
            bpp = header.bpp;
            if (bpp <= 8) {
                io_stream->tell(io_stream->instance, &palette_start);
                palette_entry_size = 3;
                ncolors = 1u << bpp;
            }
        } else if (length >= 50 && header_size >= 40) {
            BitmapInfoHeader header = ReadValue<BitmapInfoHeader>(io_stream);
            io_stream->skip(io_stream->instance, header_size - sizeof(header)); // Skip the ignored part of header
            image_info->plane_info[0].width = abs(header.width);
            image_info->plane_info[0].height = abs(header.heigth);
            bpp = header.bpp;
            compression_type = header.compression;
            ncolors = header.colors_used;
            if (bpp <= 8) {
                io_stream->tell(io_stream->instance, &palette_start);
                palette_entry_size = 4;
                ncolors = ncolors == 0 ? 1u << bpp : ncolors;
            }
        } else {
            NVIMGCDCS_LOG_ERROR("Unexpected length of a BMP header");
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }

        // sanity check
        if (palette_start != 0) { // this silences a warning about unused variable
            assert(palette_start + (ncolors * palette_entry_size) <= length);
        }

        image_info->num_planes = number_of_channels(io_stream, bpp, compression_type, ncolors, palette_entry_size);
        for (size_t p = 0; p < image_info->num_planes; p++) {
            image_info->plane_info[p].height = image_info->plane_info[0].height;
            image_info->plane_info[p].width = image_info->plane_info[0].width;
            image_info->plane_info[p].num_channels = 1;
            image_info->plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8; // TODO(janton) always?
        }
        if (image_info->num_planes == 1) {
            image_info->sample_format = NVIMGCDCS_SAMPLEFORMAT_P_Y;
            image_info->color_spec = NVIMGCDCS_COLORSPEC_GRAY;
        } else {
            image_info->sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
            image_info->color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        }
        image_info->orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
        image_info->chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;

        return NVIMGCDCS_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from bmp stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t BMPParserPlugin::Parser::static_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("bmp_parser_get_image_info");
        CHECK_NULL(parser);
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        auto handle = reinterpret_cast<BMPParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from bmp code stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

class BmpParserExtension
{
  public:
    explicit BmpParserExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
    {
        framework->registerParser(framework->instance, bmp_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
    }
    ~BmpParserExtension()
    {
         framework_->unregisterParser(framework_->instance, bmp_parser_plugin_.getParserDesc());
    }

  private:
    const nvimgcdcsFrameworkDesc_t framework_;
    BMPParserPlugin bmp_parser_plugin_;
};

nvimgcdcsStatus_t bmp_parser_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    NVIMGCDCS_LOG_TRACE("bmp_parser_extension_create");
    try {
        CHECK_NULL(framework)
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new BmpParserExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t bmp_parser_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("bmp_parser_extension_destroy");
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcdcs::BmpParserExtension*>(extension);
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t bmp_parser_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "bmp_parser_extension",
    NVIMGCDCS_VER,
    NVIMGCDCS_EXT_API_VER,

    bmp_parser_extension_create,
    bmp_parser_extension_destroy
};
// clang-format on

nvimgcdcsStatus_t get_bmp_parser_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = bmp_parser_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace nvimgcdcs