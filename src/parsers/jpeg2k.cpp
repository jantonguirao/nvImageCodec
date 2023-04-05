/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "parsers/jpeg2k.h"
#include <nvimgcodecs.h>
#include <vector>

#include "parsers/byte_io.h"
#include "parsers/exif.h"
#include "log.h"
#include "logger.h"
#include "exception.h"

// Reference: https://www.itu.int/rec/T-REC-T.800-200208-S

namespace nvimgcdcs {

namespace {

using block_type_t = std::array<uint8_t, 4>;
const block_type_t jp2_signature = {'j', 'P', ' ', ' '};      // JPEG2000 signature
const block_type_t jp2_file_type = {'f', 't', 'y', 'p'};      // File type
const block_type_t jp2_header = {'j', 'p', '2', 'h'};         // JPEG2000 header (super box)
const block_type_t jp2_image_header = {'i', 'h', 'd', 'r'};   // Image header
const block_type_t jp2_colour_spec = {'c', 'o', 'l', 'r'};     // Color specification
const block_type_t jp2_code_stream = {'j', 'p', '2', 'c'};    // Contiguous code stream
const block_type_t jp2_url = {'u', 'r', 'l', ' '};            // Data entry URL
const block_type_t jp2_palette = {'p', 'c', 'l', 'r'};        // Palette
const block_type_t jp2_cmap = {'c', 'm', 'a', 'p'};           // Component mapping
const block_type_t jp2_cdef = {'c', 'd', 'e', 'f'};           // Channel definition
const block_type_t jp2_dtbl = {'d', 't', 'b', 'l'};           // Data reference
const block_type_t jp2_bpcc = {'b', 'p', 'c', 'c'};           // Bits per component
const block_type_t jp2_jp2 = {'j', 'p', '2', ' '};            // File type fields

bool ReadBoxHeader(block_type_t& block_type, uint32_t &block_size, nvimgcdcsIoStreamDesc_t io_stream) {
    block_size = ReadValueBE<uint32_t>(io_stream);
    block_type = ReadValue<block_type_t>(io_stream);
    return true;
}

void SkipBox(nvimgcdcsIoStreamDesc_t io_stream, block_type_t expected_block, const char* box_description) {
    auto block_size = ReadValueBE<uint32_t>(io_stream);
    auto block_type = ReadValue<block_type_t>(io_stream);
    if (block_type != expected_block)
        throw std::runtime_error(std::string("Failed to read ") + std::string(box_description));
    io_stream->skip(io_stream->instance, block_size - sizeof(block_size) - sizeof(block_type));
}

}  // namespace

JPEG2KParserPlugin::JPEG2KParserPlugin()
    : parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, nullptr,
          this,              // instance
          "jpeg2k_parser",   // id
          0x00000100,        // version
          "jpeg2k",          // codec_type
          static_can_parse, static_create, Parser::static_destroy,
          Parser::static_create_parse_state, Parser::static_destroy_parse_state,
          Parser::static_get_image_info, Parser::static_get_capabilities}
{
}

struct nvimgcdcsParserDesc* JPEG2KParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::canParse(bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    io_stream->skip(io_stream->instance, sizeof(uint32_t));  // skip block size
    block_type_t block_type = ReadValue<block_type_t>(io_stream);
    *result = (block_type == jp2_signature);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::static_can_parse(
    void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg2k_parser_can_parse");
        CHECK_NULL(instance);
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
        auto handle = reinterpret_cast<JPEG2KParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not check if code stream can be parsed - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

JPEG2KParserPlugin::Parser::Parser()
{
}

nvimgcdcsStatus_t JPEG2KParserPlugin::create(nvimgcdcsParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcdcsParser_t>(
        new JPEG2KParserPlugin::Parser());
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg2k_parser_create");
        CHECK_NULL(instance);
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<JPEG2KParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not create jpeg parser - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg2k_parser_destroy");
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<JPEG2KParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not destroy jpeg parser - " << e.what());
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::getCapabilities(
    const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    // TODO(janton): remove this API
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_get_capabilities(
    nvimgcdcsParser_t parser, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg_get_capabilities");
        CHECK_NULL(parser);
        CHECK_NULL(capabilities);
        CHECK_NULL(size);
        auto handle = reinterpret_cast<JPEG2KParserPlugin::Parser*>(parser);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrive jpeg parser capabilites - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::createParseState(nvimgcdcsParseState_t* parse_state)
{
    // TODO(janton): remove this API
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_create_parse_state(
    nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state)
{
    try {
        NVIMGCDCS_LOG_TRACE("JPEG create_parse_state");
        CHECK_NULL(parser);
        CHECK_NULL(parse_state);
        auto handle = reinterpret_cast<JPEG2KParserPlugin::Parser*>(parser);
        return handle->createParseState(parse_state);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not create jpeg parse state - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_destroy_parse_state(
    nvimgcdcsParseState_t parse_state)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg_destroy_parse_state");
        CHECK_NULL(parse_state);
        // TODO(janton): remove this API
        return NVIMGCDCS_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not destroy jpeg parse state - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}


nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::getImageInfo(
    nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    NVIMGCDCS_LOG_TRACE("jpeg2k_parser_get_image_info");
    try {
        size_t stream_size = 0;
        nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
        io_stream->size(io_stream->instance, &stream_size);
        io_stream->seek(io_stream->instance, 0, SEEK_SET);

        SkipBox(io_stream, jp2_signature, "JPEG2K signature");
        SkipBox(io_stream, jp2_file_type, "JPEG2K file type");

        uint32_t block_size;
        block_type_t block_type;
        uint16_t num_components = 0;
        uint32_t height = 0, width = 0;
        uint8_t bits_per_component = 8;
        uint8_t sign_component = 0;
        nvimgcdcsChromaSubsampling_t subsampling;
        nvimgcdcsColorSpec_t color_spec = NVIMGCDCS_COLORSPEC_UNKNOWN;
        while (ReadBoxHeader(block_type, block_size, io_stream)) {
            if (block_type == jp2_header) {  // superbox
                auto remaining_bytes = block_size - sizeof(block_size) - sizeof(block_type);
                while (remaining_bytes > 0) {
                    ReadBoxHeader(block_type, block_size, io_stream);
                    if (block_type == jp2_image_header) {
                        if (block_size != 22) {
                            NVIMGCDCS_LOG_ERROR("Invalid JPEG2K image header");
                            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
                        }
                        height = ReadValueBE<uint32_t>(io_stream);
                        width = ReadValueBE<uint32_t>(io_stream);
                        num_components = ReadValueBE<uint16_t>(io_stream);
                        bits_per_component = ReadValueBE<uint8_t>(io_stream);
                        sign_component = bits_per_component >> 7;
                        bits_per_component = bits_per_component & 0x7f;
                        bits_per_component += 1;
                        io_stream->skip(io_stream->instance, sizeof(uint8_t));  // compression_type
                        io_stream->skip(io_stream->instance, sizeof(uint8_t));  // color_space_unknown
                        io_stream->skip(io_stream->instance, sizeof(uint8_t));  // IPR
                    } else if (block_type == jp2_colour_spec && color_spec == NVIMGCDCS_COLORSPEC_UNKNOWN) {
                        auto method = ReadValueBE<uint8_t>(io_stream);
                        io_stream->skip(io_stream->instance, sizeof(int8_t));  // precedence
                        io_stream->skip(io_stream->instance, sizeof(int8_t));  // colourspace approximation
                        auto enumCS = ReadValueBE<uint32_t>(io_stream);
                        if (method == 1) {
                            switch (enumCS) {
                                case 16:  // sRGB
                                    color_spec = NVIMGCDCS_COLORSPEC_SRGB;
                                    break;
                                case 17:  // Greyscale
                                    color_spec = NVIMGCDCS_COLORSPEC_GRAY;
                                    break;
                                case 18:
                                    color_spec = NVIMGCDCS_COLORSPEC_SYCC;
                                    break;
                                default:
                                    color_spec = NVIMGCDCS_COLORSPEC_UNSUPPORTED;
                                    break;
                            }
                        } else if (method == 2) {
                            color_spec = NVIMGCDCS_COLORSPEC_UNSUPPORTED;
                        }
                    } else {
                        io_stream->skip(io_stream->instance, block_size - sizeof(block_size) - sizeof(block_type));
                    }
                    remaining_bytes -= block_size;
                }
            } else if (block_type == jp2_code_stream) {
                auto marker = ReadValueBE<uint16_t>(io_stream);
                if(marker != 0xFF4F) {  // SOC marker
                    NVIMGCDCS_LOG_ERROR("SOC marker not found");
                    return NVIMGCDCS_STATUS_BAD_CODESTREAM;
                }
                // SOC should be followed by SIZ. Figure A.3
                marker = ReadValueBE<uint16_t>(io_stream);
                if (marker != 0xFF51) { // SIZ marker
                    NVIMGCDCS_LOG_ERROR("SIZ marker not found");
                    return NVIMGCDCS_STATUS_BAD_CODESTREAM;
                }

                auto marker_size = ReadValueBE<uint16_t>(io_stream);
                if(marker_size < 41 || marker_size > 49190) {
                    NVIMGCDCS_LOG_ERROR("Invalid SIZ marker size");
                    return NVIMGCDCS_STATUS_BAD_CODESTREAM;
                }
                io_stream->skip(io_stream->instance, sizeof(uint16_t));  // RSiz
                io_stream->skip(io_stream->instance, sizeof(uint32_t));  // XSiz
                io_stream->skip(io_stream->instance, sizeof(uint32_t));  // YSiz
                io_stream->skip(io_stream->instance, sizeof(uint32_t));  // XOSiz
                io_stream->skip(io_stream->instance, sizeof(uint32_t));  // YOSiz
                io_stream->skip(io_stream->instance, sizeof(uint32_t));  // XTSiz
                io_stream->skip(io_stream->instance, sizeof(uint32_t));  // YTSiz
                io_stream->skip(io_stream->instance, sizeof(uint32_t));  // XTOSiz
                io_stream->skip(io_stream->instance, sizeof(uint32_t));  // YTOSiz

                uint16_t CSiz = ReadValueBE<uint16_t>(io_stream);

                // CSiz in table A.9, minimum of 1 and Max of 16384
                if (CSiz < 1 || CSiz > 16384) {
                    NVIMGCDCS_LOG_ERROR("Invalid number of components");
                    return NVIMGCDCS_STATUS_BAD_CODESTREAM;
                }

                std::vector<uint8_t> XRSiz(CSiz);
                std::vector<uint8_t> YRSiz(CSiz);
                for (int i = 0; i < CSiz; i++) {
                    io_stream->skip(io_stream->instance, sizeof(uint8_t));  // precision
                    XRSiz[i] = ReadValue<uint8_t>(io_stream);
                    YRSiz[i] = ReadValue<uint8_t>(io_stream);
                }
                if(CSiz == 3 || CSiz == 4) {
                    if ((XRSiz[0] == 1) && (XRSiz[1] == 2) && (XRSiz[2] == 2)
                     && (YRSiz[0] == 1) && (YRSiz[1] == 2) && (YRSiz[2] == 2)) {
                        subsampling = NVIMGCDCS_SAMPLING_420;
                    } else if ((XRSiz[0] == 1) && (XRSiz[1] == 2) && (XRSiz[2] == 2)
                            && (YRSiz[0] == 1) && (YRSiz[1] == 1) && (YRSiz[2] == 1)) {
                        subsampling = NVIMGCDCS_SAMPLING_422;
                    } else if ((XRSiz[0] == 1) && (XRSiz[1] == 1) && (XRSiz[2] == 1)
                            && (YRSiz[0] == 1) && (YRSiz[1] == 1) && (YRSiz[2] == 1)) {
                        subsampling = NVIMGCDCS_SAMPLING_444;
                    } else {
                        NVIMGCDCS_LOG_ERROR("Unsupported chroma subsampling");
                        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
                    }
                }
                break;  // stop parsing here
            }
        }

        image_info->type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
        image_info->sample_format = num_components > 1 ? NVIMGCDCS_SAMPLEFORMAT_P_RGB : NVIMGCDCS_SAMPLEFORMAT_P_Y;
        image_info->orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
        image_info->chroma_subsampling = subsampling;
        image_info->color_spec = color_spec;
        image_info->num_planes = num_components;
        auto sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        if (bits_per_component <= 16 && bits_per_component > 8) {
            sample_type = sign_component ?
                NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16 :
                NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
        } else if (bits_per_component <= 8) {
            sample_type = sign_component ?
                NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8 :
                NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        } else {
            NVIMGCDCS_LOG_ERROR(
                "Not supported precision " << bits_per_component << " (sign=" <<
                (int)sign_component << ")");
            return NVIMGCDCS_STATUS_INTERNAL_ERROR;
        }
        for (int p = 0; p < num_components; p++) {
            image_info->plane_info[p].height = height;
            image_info->plane_info[p].width = width;
            image_info->plane_info[p].num_channels = 1;
            image_info->plane_info[p].sample_type = sample_type;
        }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from jpeg stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_get_image_info(nvimgcdcsParser_t parser,
    nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg2k_parser_get_image_info");
        CHECK_NULL(parser);
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        auto handle = reinterpret_cast<JPEG2KParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from jpeg code stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

static auto jpeg2k_parser_plugin = JPEG2KParserPlugin();

nvimgcdcsStatus_t jpeg2k_parser_extension_create(
    const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t* extension)
{
    NVIMGCDCS_LOG_TRACE("extension_create");

    framework->registerParser(framework->instance, jpeg2k_parser_plugin.getParserDesc());

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t jpeg2k_parser_extension_destroy(
    const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("jpeg2k_parser_extension_destroy");

    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t jpeg2k_parser_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    "jpeg2k_parser_extension",  // id
     0x00000100,              // version

    jpeg2k_parser_extension_create,
    jpeg2k_parser_extension_destroy
};
// clang-format on

nvimgcdcsStatus_t get_jpeg2k_parser_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = jpeg2k_parser_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}

}  // namespace nvimgcdcs