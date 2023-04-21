/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "parsers/pnm.h"
#include <nvimgcodecs.h>
#include <vector>

#include "exception.h"
#include "log.h"
#include "logger.h"
#include "parsers/byte_io.h"
#include "parsers/exif.h"
#include "exif_orientation.h"

namespace nvimgcdcs {

namespace {

// comments can appear in the middle of tokens, and the newline at the
// end is part of the comment, not counted as whitespace
// http://netpbm.sourceforge.net/doc/pbm.html
size_t SkipComment(nvimgcdcsIoStreamDesc_t io_stream)
{
    char c;
    size_t skipped = 0;
    do {
        c = ReadValue<char>(io_stream);
        skipped++;
    } while (c != '\n');
    return skipped;
}

void SkipSpaces(nvimgcdcsIoStreamDesc_t io_stream)
{
    size_t pos;
    io_stream->tell(io_stream->instance, &pos);
    while (true) {
        char c = ReadValue<char>(io_stream);
        pos++;
        if (c == '#')
            pos += SkipComment(io_stream);
        else if (!isspace(c))
            break;
    }
    // return the nonspace byte to the stream
    io_stream->seek(io_stream->instance, pos-1, SEEK_SET);
}

int ParseInt(nvimgcdcsIoStreamDesc_t io_stream)
{
    size_t pos;
    io_stream->tell(io_stream->instance, &pos);
    int int_value = 0;
    while (true) {
        char c = ReadValue<char>(io_stream);
        pos++;
        if (isdigit(c))
            int_value = int_value * 10 + (c - '0');
        else if (c == '#')
            pos += SkipComment(io_stream);
        else
            break;
    }
    // return the nondigit byte to the stream
    io_stream->seek(io_stream->instance, pos-1, SEEK_SET);
    return int_value;
}

nvimgcdcsStatus_t GetImageInfoImpl(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    size_t io_stream_length;
    io_stream->size(io_stream->instance, &io_stream_length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);

    if (image_info->type != NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO) {
        NVIMGCDCS_LOG_ERROR("Unexpected structure type");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    
    // http://netpbm.sourceforge.net/doc/ppm.html

    if (io_stream_length < 3) {
        NVIMGCDCS_LOG_ERROR("Unexpected end of stream");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }
    
    std::array<uint8_t, 3> header = ReadValue<std::array<uint8_t, 3>>(io_stream);
    bool is_pnm = header[0] == 'P' && header[1] >= '1' && header[1] <= '6' && isspace(header[2]);
    if (!is_pnm) {
        NVIMGCDCS_LOG_ERROR("Unexpected header");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }
    // formats "P3" and "P6" are RGB color, all other formats are bitmaps or greymaps
    uint32_t nchannels = (header[1] == '3' || header[1] == '6') ? 3 : 1;

    SkipSpaces(io_stream);
    uint32_t width = ParseInt(io_stream);
    SkipSpaces(io_stream);
    uint32_t height = ParseInt(io_stream);

    image_info->sample_format = nchannels >= 3 ? NVIMGCDCS_SAMPLEFORMAT_P_RGB : NVIMGCDCS_SAMPLEFORMAT_P_Y;
    image_info->orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
    image_info->chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;
    image_info->color_spec = NVIMGCDCS_COLORSPEC_SRGB;
    image_info->num_planes = nchannels;
    for (size_t p = 0; p < nchannels; p++) {
        image_info->plane_info[p].height = height;
        image_info->plane_info[p].width = width;
        image_info->plane_info[p].num_channels = 1;
        image_info->plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace

PNMParserPlugin::PNMParserPlugin()
    : parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, nullptr,
          this,          // instance
          "pnm_parser", // id
          0x00000100,    // version
          "pnm",        // codec_type
          static_can_parse, static_create, Parser::static_destroy, Parser::static_create_parse_state, Parser::static_destroy_parse_state,
          Parser::static_get_image_info, Parser::static_get_capabilities}
{}

struct nvimgcdcsParserDesc* PNMParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t PNMParserPlugin::canParse(bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    size_t length;
    io_stream->size(io_stream->instance, &length);
    if (length < 3) {
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }
    std::array<uint8_t, 3> header = ReadValue<std::array<uint8_t, 3>>(io_stream);
    *result = header[0] == 'P' && header[1] >= '1' && header[1] <= '6' && isspace(header[2]);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNMParserPlugin::static_can_parse(void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("pnm_parser_can_parse");
        CHECK_NULL(instance);
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
        auto handle = reinterpret_cast<PNMParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not check if code stream can be parsed - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

PNMParserPlugin::Parser::Parser()
{}

nvimgcdcsStatus_t PNMParserPlugin::create(nvimgcdcsParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcdcsParser_t>(new PNMParserPlugin::Parser());
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNMParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("pnm_parser_create");
        CHECK_NULL(instance);
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNMParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not create pnm parser - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNMParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("pnm_parser_destroy");
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNMParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not destroy pnm parser - " << e.what());
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNMParserPlugin::Parser::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t PNMParserPlugin::Parser::static_get_capabilities(
    nvimgcdcsParser_t parser, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_LOG_TRACE("pnm_get_capabilities");
        CHECK_NULL(parser);
        CHECK_NULL(capabilities);
        CHECK_NULL(size);
        auto handle = reinterpret_cast<PNMParserPlugin::Parser*>(parser);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve pnm parser capabilites - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t PNMParserPlugin::Parser::createParseState(nvimgcdcsParseState_t* parse_state)
{
    // TODO(janton): remove this API
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNMParserPlugin::Parser::static_create_parse_state(nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state)
{
    try {
        NVIMGCDCS_LOG_TRACE("JPEG create_parse_state");
        CHECK_NULL(parser);
        CHECK_NULL(parse_state);
        auto handle = reinterpret_cast<PNMParserPlugin::Parser*>(parser);
        return handle->createParseState(parse_state);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not create pnm parse state - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t PNMParserPlugin::Parser::static_destroy_parse_state(nvimgcdcsParseState_t parse_state)
{
    try {
        NVIMGCDCS_LOG_TRACE("pnm_destroy_parse_state");
        CHECK_NULL(parse_state);
        // TODO(janton): remove this API
        return NVIMGCDCS_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not destroy pnm parse state - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t PNMParserPlugin::Parser::getImageInfo(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    NVIMGCDCS_LOG_TRACE("pnm_parser_get_image_info");
    try {
        return GetImageInfoImpl(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from png stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    }
}

nvimgcdcsStatus_t PNMParserPlugin::Parser::static_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("pnm_parser_get_image_info");
        CHECK_NULL(parser);
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        auto handle = reinterpret_cast<PNMParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from pnm code stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

static auto pnm_parser_plugin = PNMParserPlugin();

nvimgcdcsStatus_t pnm_parser_extension_create(const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t* extension)
{
    NVIMGCDCS_LOG_TRACE("extension_create");

    framework->registerParser(framework->instance, pnm_parser_plugin.getParserDesc());

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t pnm_parser_extension_destroy(const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("pnm_parser_extension_destroy");

    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t pnm_parser_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    "pnm_parser_extension",  // id
     0x00000100,             // version

    pnm_parser_extension_create,
    pnm_parser_extension_destroy
};
// clang-format on

nvimgcdcsStatus_t get_pnm_parser_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = pnm_parser_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace nvimgcdcs