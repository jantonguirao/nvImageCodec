/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "parsers/png.h"
#include <nvimgcodecs.h>
#include <vector>

#include "exception.h"
#include "exif_orientation.h"
#include "log.h"
#include "logger.h"
#include "parsers/byte_io.h"
#include "parsers/exif.h"

namespace nvimgcdcs {

namespace {

// https://www.w3.org/TR/2003/REC-PNG-20031110

enum ColorType : uint8_t
{
    PNG_COLOR_TYPE_GRAY = 0,
    PNG_COLOR_TYPE_RGB = 2,
    PNG_COLOR_TYPE_PALETTE = 3,
    PNG_COLOR_TYPE_GRAY_ALPHA = 4,
    PNG_COLOR_TYPE_RGBA = 6
};

struct IhdrChunk
{
    uint32_t width;
    uint32_t height;
    uint8_t color_type;
    // Some fields were ommited.

    int GetNumberOfChannels(bool include_alpha)
    {
        switch (color_type) {
        case PNG_COLOR_TYPE_GRAY:
            return 1;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            return 1 + include_alpha;
        case PNG_COLOR_TYPE_RGB:
        case PNG_COLOR_TYPE_PALETTE: // 1 byte but it's converted to 3-channel BGR by OpenCV
            return 3;
        case PNG_COLOR_TYPE_RGBA:
            return 3 + include_alpha;
        default:
            throw std::runtime_error("color type not supported");
        }
    }
};

using chunk_type_field_t = std::array<uint8_t, 4>;
static constexpr chunk_type_field_t IHDR_TAG{'I', 'H', 'D', 'R'};
static constexpr chunk_type_field_t EXIF_TAG{'e', 'X', 'I', 'f'};
static constexpr chunk_type_field_t IEND_TAG{'I', 'E', 'N', 'D'};

using png_signature_t = std::array<uint8_t, 8>;
static constexpr png_signature_t PNG_SIGNATURE = {137, 80, 78, 71, 13, 10, 26, 10};

nvimgcdcsStatus_t GetImageInfoImpl(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{

    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    size_t io_stream_length;
    io_stream->size(io_stream->instance, &io_stream_length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);

    size_t read_nbytes = 0;
    png_signature_t signature;
    io_stream->read(io_stream->instance, &read_nbytes, &signature[0], signature.size());
    if (read_nbytes != sizeof(png_signature_t) || signature != PNG_SIGNATURE) {
        NVIMGCDCS_LOG_ERROR("Unexpected signature");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    // IHDR Chunk:
    // IHDR chunk length(4 bytes): 0x00 0x00 0x00 0x0D
    // IHDR chunk type(Identifies chunk type to be IHDR): 0x49 0x48 0x44 0x52
    // Image width in pixels(variable 4): xx xx xx xx
    // Image height in pixels(variable 4): xx xx xx xx
    // Flags in the chunk(variable 5 bytes): xx xx xx xx xx
    // CRC checksum(variable 4 bytes): xx xx xx xx

    uint32_t length = ReadValueBE<uint32_t>(io_stream);
    if (length != (4 + 4 + 5)) {
        NVIMGCDCS_LOG_ERROR("Unexpected length");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    auto chunk_type = ReadValue<chunk_type_field_t>(io_stream);
    if (chunk_type != IHDR_TAG) {
        NVIMGCDCS_LOG_ERROR("Missing IHDR tag");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    if (image_info->type != NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO) {
        NVIMGCDCS_LOG_ERROR("Unexpected structure type");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    image_info->chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;
    image_info->plane_info[0].width = ReadValueBE<uint32_t>(io_stream);
    image_info->plane_info[0].height = ReadValueBE<uint32_t>(io_stream);
    uint8_t bitdepth = ReadValueBE<uint8_t>(io_stream);
    nvimgcdcsSampleDataType_t sample_format;
    switch (bitdepth) {
    case 1:
    case 2:
    case 4:
    case 8:
        sample_format = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        break;
    case 16:
        sample_format = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
        break;
    default:
        NVIMGCDCS_LOG_ERROR("Unexpected bitdepth: " << bitdepth);
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }
    // see Table 11.1
    auto color_type = static_cast<ColorType>(ReadValueBE<uint8_t>(io_stream));
    switch (color_type) {
    case PNG_COLOR_TYPE_GRAY:
        image_info->num_planes = 1;
        break;
    case PNG_COLOR_TYPE_RGB:
        image_info->num_planes = 3;
        if (bitdepth != 8 && bitdepth != 16) {
            NVIMGCDCS_LOG_ERROR("Unexpected bitdepth for RGB color type: " << bitdepth);
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }
        break;
    case PNG_COLOR_TYPE_PALETTE:
        image_info->num_planes = 3;
        if (bitdepth == 16) {
            NVIMGCDCS_LOG_ERROR("Unexpected bitdepth for palette color type: " << bitdepth);
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }
        break;
    case PNG_COLOR_TYPE_GRAY_ALPHA:
        image_info->num_planes = 2;
        if (bitdepth != 8 && bitdepth != 16) {
            NVIMGCDCS_LOG_ERROR("Unexpected bitdepth for Gray with alpha color type: " << bitdepth);
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }
        break;
    case PNG_COLOR_TYPE_RGBA:
        image_info->num_planes = 4;
        if (bitdepth != 8 && bitdepth != 16) {
            NVIMGCDCS_LOG_ERROR("Unexpected bitdepth for RGBA color type: " << bitdepth);
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }
        break;
    default:
        NVIMGCDCS_LOG_ERROR("Unexpected color type: " << color_type);
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }
    image_info->sample_format = image_info->num_planes >= 3 ? NVIMGCDCS_SAMPLEFORMAT_P_RGB : NVIMGCDCS_SAMPLEFORMAT_P_Y;
    image_info->color_spec = image_info->num_planes >= 3 ? NVIMGCDCS_COLORSPEC_SRGB : NVIMGCDCS_COLORSPEC_GRAY;

    for (size_t p = 0; p < image_info->num_planes; p++) {
        image_info->plane_info[p].height = image_info->plane_info[0].height;
        image_info->plane_info[p].width = image_info->plane_info[0].width;
        image_info->plane_info[p].num_channels = 1;
        image_info->plane_info[p].sample_type = sample_format;
    }

    io_stream->skip(io_stream->instance, 3 + 4); // Skip the other fields and the CRC checksum.
    image_info->orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
    while (true) {
        uint32_t chunk_length = ReadValueBE<uint32_t>(io_stream);
        auto chunk_type = ReadValue<chunk_type_field_t>(io_stream);

        if (chunk_type == IEND_TAG)
            break;

        if (chunk_type == EXIF_TAG) {
            std::vector<uint8_t> chunk(chunk_length);
            size_t read_chunk_nbytes;
            io_stream->read(io_stream->instance, &read_chunk_nbytes, &chunk[0], chunk_length);
            if (read_chunk_nbytes != chunk_length) {
                NVIMGCDCS_LOG_ERROR("Unexpected end of stream");
                return NVIMGCDCS_STATUS_BAD_CODESTREAM;
            }

            cv::ExifReader reader;
            if (reader.parseExif(chunk.data(), chunk.size())) {
                auto entry = reader.getTag(cv::ORIENTATION);
                if (entry.tag != cv::INVALID_TAG) {
                    image_info->orientation = FromExifOrientation(static_cast<ExifOrientation>(entry.field_u16));
                }
            }
            io_stream->skip(io_stream->instance, 4);                // 4 bytes of CRC
        } else {
            io_stream->skip(io_stream->instance, chunk_length + 4); // + 4 bytes of CRC
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // end namespace

PNGParserPlugin::PNGParserPlugin()
    : parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, nullptr,
          this,         // instance
          "png_parser", // id
          0x00000100,   // version
          "png",        // codec_type
          static_can_parse, static_create, Parser::static_destroy, Parser::static_get_image_info, Parser::static_get_capabilities}
{
}

struct nvimgcdcsParserDesc* PNGParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t PNGParserPlugin::canParse(bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    size_t read_nbytes = 0;
    png_signature_t signature;
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    io_stream->read(io_stream->instance, &read_nbytes, &signature[0], signature.size());
    if (read_nbytes == sizeof(png_signature_t) && signature == PNG_SIGNATURE) {
        *result = true;
    } else {
        *result = false;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNGParserPlugin::static_can_parse(void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("png_parser_can_parse");
        CHECK_NULL(instance);
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
        auto handle = reinterpret_cast<PNGParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not check if code stream can be parsed - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

PNGParserPlugin::Parser::Parser()
{
}

nvimgcdcsStatus_t PNGParserPlugin::create(nvimgcdcsParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcdcsParser_t>(new PNGParserPlugin::Parser());
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNGParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("png_parser_create");
        CHECK_NULL(instance);
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNGParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not create png parser - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNGParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("png_parser_destroy");
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNGParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not destroy png parser - " << e.what());
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNGParserPlugin::Parser::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    if (capabilities) {
        *capabilities = capabilities_.data();
    }

    if (size) {
        *size = capabilities_.size();
    } else {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNGParserPlugin::Parser::static_get_capabilities(
    nvimgcdcsParser_t parser, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_LOG_TRACE("png_get_capabilities");
        CHECK_NULL(parser);
        CHECK_NULL(capabilities);
        CHECK_NULL(size);
        auto handle = reinterpret_cast<PNGParserPlugin::Parser*>(parser);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve png parser capabilites - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t PNGParserPlugin::Parser::getImageInfo(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    NVIMGCDCS_LOG_TRACE("png_parser_get_image_info");
    try {
        return GetImageInfoImpl(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from png stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNGParserPlugin::Parser::static_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("png_parser_get_image_info");
        CHECK_NULL(parser);
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        auto handle = reinterpret_cast<PNGParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from png code stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

class PngParserExtension
{
  public:
    explicit PngParserExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
    {
        framework->registerParser(framework->instance, png_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
    }
    ~PngParserExtension() { framework_->unregisterParser(framework_->instance, png_parser_plugin_.getParserDesc()); }

  private:
    const nvimgcdcsFrameworkDesc_t framework_;
    PNGParserPlugin png_parser_plugin_;
};

nvimgcdcsStatus_t png_parser_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    NVIMGCDCS_LOG_TRACE("png_parser_extension_create");
    try {
        CHECK_NULL(framework)
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new PngParserExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t png_parser_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("png_parser_extension_destroy");
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcdcs::PngParserExtension*>(extension);
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t png_parser_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "png_parser_extension",  // id
     0x00000100,             // version

    png_parser_extension_create,
    png_parser_extension_destroy
};
// clang-format on

nvimgcdcsStatus_t get_png_parser_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = png_parser_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace nvimgcdcs