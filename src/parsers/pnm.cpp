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
#include <string.h>
#include <vector>

#include "exception.h"
#include "exif_orientation.h"
#include "log_ext.h"

#include "parsers/byte_io.h"
#include "parsers/exif.h"

namespace nvimgcdcs {

namespace {

// comments can appear in the middle of tokens, and the newline at the
// end is part of the comment, not counted as whitespace
// http://netpbm.sourceforge.net/doc/pbm.html
size_t SkipComment(nvimgcdcsIoStreamDesc_t* io_stream)
{
    char c;
    size_t skipped = 0;
    do {
        c = ReadValue<char>(io_stream);
        skipped++;
    } while (c != '\n');
    return skipped;
}

void SkipSpaces(nvimgcdcsIoStreamDesc_t* io_stream)
{
    ptrdiff_t pos;
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
    io_stream->seek(io_stream->instance, pos - 1, SEEK_SET);
}

int ParseInt(nvimgcdcsIoStreamDesc_t* io_stream)
{
    ptrdiff_t pos;
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
    io_stream->seek(io_stream->instance, pos - 1, SEEK_SET);
    return int_value;
}

nvimgcdcsStatus_t GetImageInfoImpl(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t io_stream_length;
    io_stream->size(io_stream->instance, &io_stream_length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);

    if (image_info->type != NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO) {
        NVIMGCDCS_LOG_ERROR(framework, plugin_id, "Unexpected structure type");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    strcpy(image_info->codec_name, "pnm");
    // http://netpbm.sourceforge.net/doc/ppm.html

    if (io_stream_length < 3) {
        NVIMGCDCS_LOG_ERROR(framework, plugin_id, "Unexpected end of stream");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    std::array<uint8_t, 3> header = ReadValue<std::array<uint8_t, 3>>(io_stream);
    bool is_pnm = header[0] == 'P' && header[1] >= '1' && header[1] <= '6' && isspace(header[2]);
    if (!is_pnm) {
        NVIMGCDCS_LOG_ERROR(framework, plugin_id, "Unexpected header");
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

PNMParserPlugin::PNMParserPlugin(const nvimgcdcsFrameworkDesc_t* framework)
    : framework_(framework)
    , parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, nullptr, this, plugin_id_, "pnm", static_can_parse, static_create,
          Parser::static_destroy, Parser::static_get_image_info}
{
}

nvimgcdcsParserDesc_t* PNMParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t PNMParserPlugin::canParse(int* result, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "pnm_parser_can_parse");
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
    nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t length;
    io_stream->size(io_stream->instance, &length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    if (length < 3) {
        *result = 0;
        return NVIMGCDCS_STATUS_SUCCESS;
    }
    std::array<uint8_t, 3> header = ReadValue<std::array<uint8_t, 3>>(io_stream);
    *result = header[0] == 'P' && header[1] >= '1' && header[1] <= '6' && isspace(header[2]);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not check if code stream can be parsed - " << e.what());
        return NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNMParserPlugin::static_can_parse(void* instance, int* result, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<PNMParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

PNMParserPlugin::Parser::Parser(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework)
    : plugin_id_(plugin_id)
    , framework_(framework)
{
    NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "pnm_parser_destroy");
}

nvimgcdcsStatus_t PNMParserPlugin::create(nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "pnm_parser_create");
        CHECK_NULL(parser);
        *parser = reinterpret_cast<nvimgcdcsParser_t>(new PNMParserPlugin::Parser(plugin_id_, framework_));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not create pnm parser - " << e.what());
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNMParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<PNMParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNMParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNMParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PNMParserPlugin::Parser::getImageInfo(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "pnm_parser_get_image_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        return GetImageInfoImpl(plugin_id_, framework_, image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from png stream - " << e.what());
        return NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcdcsStatus_t PNMParserPlugin::Parser::static_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNMParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

class PnmParserExtension
{
  public:
    explicit PnmParserExtension(const nvimgcdcsFrameworkDesc_t* framework)
        : framework_(framework)
        , pnm_parser_plugin_(framework)        
    {
        framework->registerParser(framework->instance, pnm_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
    }
    ~PnmParserExtension() { framework_->unregisterParser(framework_->instance, pnm_parser_plugin_.getParserDesc()); }

    static nvimgcdcsStatus_t pnm_parser_extension_create(
        void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework)
{
    try {
        CHECK_NULL(framework)
            NVIMGCDCS_LOG_TRACE(framework, "pnm_parser_ext", "pnm_parser_extension_create");
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new PnmParserExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

    static nvimgcdcsStatus_t pnm_parser_extension_destroy(nvimgcdcsExtension_t extension)
{
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcdcs::PnmParserExtension*>(extension);
            NVIMGCDCS_LOG_TRACE(ext_handle->framework_, "pnm_parser_ext", "pnm_parser_extension_destroy");
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

  private:
    const nvimgcdcsFrameworkDesc_t* framework_;
    PNMParserPlugin pnm_parser_plugin_;
};

// clang-format off
nvimgcdcsExtensionDesc_t pnm_parser_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "pnm_parser_extension",
    NVIMGCDCS_VER,
    NVIMGCDCS_EXT_API_VER,

    PnmParserExtension::pnm_parser_extension_create,
    PnmParserExtension::pnm_parser_extension_destroy
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