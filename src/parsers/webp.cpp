/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "parsers/webp.h"
#include <nvimgcodecs.h>
#include <vector>
#include <string.h>

#include "exception.h"
#include "exif_orientation.h"
#include "log.h"
#include "logger.h"
#include "parsers/byte_io.h"
#include "parsers/exif.h"

namespace nvimgcdcs {

namespace {

// Specific bits in WebpExtendedHeader::layout_mask
static constexpr uint8_t EXTENDED_LAYOUT_RESERVED = 1 << 0;
static constexpr uint8_t EXTENDED_LAYOUT_ANIMATION = 1 << 1;
static constexpr uint8_t EXTENDED_LAYOUT_XMP_METADATA = 1 << 2;
static constexpr uint8_t EXTENDED_LAYOUT_EXIF_METADATA = 1 << 3;
static constexpr uint8_t EXTENDED_LAYOUT_ALPHA = 1 << 4;
static constexpr uint8_t EXTENDED_LAYOUT_ICC_PROFILE = 1 << 5;

using chunk_type_t = std::array<uint8_t, 4>;
static constexpr chunk_type_t RIFF_TAG = {'R', 'I', 'F', 'F'};
static constexpr chunk_type_t WEBP_TAG = {'W', 'E', 'B', 'P'};
static constexpr chunk_type_t VP8_TAG = {'V', 'P', '8', ' '};  // lossy
static constexpr chunk_type_t VP8L_TAG = {'V', 'P', '8', 'L'}; // lossless
static constexpr chunk_type_t VP8X_TAG = {'V', 'P', '8', 'X'}; // extended
static constexpr chunk_type_t EXIF_TAG = {'E', 'X', 'I', 'F'}; // EXIF

nvimgcdcsStatus_t GetImageInfoImpl(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t io_stream_length;
    io_stream->size(io_stream->instance, &io_stream_length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);

    if (image_info->type != NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO) {
        NVIMGCDCS_LOG_ERROR(Logger::get(), "Unexpected structure type");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    strcpy(image_info->codec_name, "webp");

    if (io_stream_length < (4 + 4 + 4)) { // RIFF + file size + WEBP
        NVIMGCDCS_LOG_ERROR(Logger::get(), "Unexpected end of stream");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    // https://developers.google.com/speed/webp/docs/riff_container#webp_file_header
    auto riff = ReadValue<chunk_type_t>(io_stream);
    if (riff != RIFF_TAG) {
        NVIMGCDCS_LOG_ERROR(Logger::get(), "Unexpected RIFF tag");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }
    io_stream->skip(io_stream->instance, sizeof(uint32_t)); // file_size
    auto webp = ReadValue<chunk_type_t>(io_stream);
    if (webp != WEBP_TAG) {
        NVIMGCDCS_LOG_ERROR(Logger::get(), "Unexpected WEBP tag");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    auto chunk_type = ReadValue<chunk_type_t>(io_stream);
    auto chunk_size = ReadValueLE<uint32_t>(io_stream);
    uint32_t width = 0, height = 0, nchannels = 3;
    bool alpha = false;
    nvimgcdcsOrientation_t orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
    const uint16_t mask = (1 << 14) - 1;
    if (chunk_type == VP8_TAG) {                 // lossy format
        io_stream->skip(io_stream->instance, 3); // frame_tag
        const std::array<uint8_t, 3> expected_sync_code{0x9D, 0x01, 0x2A};
        auto sync_code = ReadValue<std::array<uint8_t, 3>>(io_stream);
        if (sync_code != expected_sync_code) {
            NVIMGCDCS_LOG_ERROR(Logger::get(), "Unexpected VP8 sync code");
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }
        width = ReadValueLE<uint16_t>(io_stream) & mask;
        height = ReadValueLE<uint16_t>(io_stream) & mask;
    } else if (chunk_type == VP8L_TAG) { // lossless format
        auto signature_byte = ReadValue<uint8_t>(io_stream);
        const uint8_t expected_signature_byte = 0x2F;
        if (signature_byte != expected_signature_byte) {
            NVIMGCDCS_LOG_ERROR(Logger::get(), "Unexpected VP8L signature byte");
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }
        auto features = ReadValueLE<uint32_t>(io_stream);
        // VP8L shape information are packed inside the features field
        width = (features & mask) + 1;
        height = ((features >> 14) & mask) + 1;
        alpha = features & (1 << (2 * 14));
    } else if (chunk_type == VP8X_TAG) { // extended format
        size_t curr, end_of_chunk_pos;
        io_stream->tell(io_stream->instance, &curr);
        end_of_chunk_pos = curr + chunk_size;

        auto layout_mask = ReadValue<uint8_t>(io_stream);
        io_stream->skip(io_stream->instance, 3); // reserved
        // Both dimensions are encoded with 24 bits, as (width - 1) i (height - 1) respectively
        width = ReadValueLE<uint32_t, 3>(io_stream) + 1;
        height = ReadValueLE<uint32_t, 3>(io_stream) + 1;
        alpha = layout_mask & EXTENDED_LAYOUT_ALPHA;
        io_stream->seek(io_stream->instance, end_of_chunk_pos, SEEK_SET);
        if (layout_mask & EXTENDED_LAYOUT_EXIF_METADATA) {
            bool exif_parsed = false;
            while (!exif_parsed) {
                chunk_type = ReadValue<chunk_type_t>(io_stream);
                chunk_size = ReadValueLE<uint32_t>(io_stream);
                io_stream->tell(io_stream->instance, &curr);
                end_of_chunk_pos = curr + chunk_size;
                if (chunk_type == EXIF_TAG) {
                    // Parse the chunk data into the orientation
                    std::vector<uint8_t> buffer(chunk_size);
                    size_t read_nbytes = 0;
                    io_stream->read(io_stream->instance, &read_nbytes, buffer.data(), buffer.size());
                    if (read_nbytes != chunk_size) {
                        NVIMGCDCS_LOG_ERROR(Logger::get(), "Unexpected end of stream");
                        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
                    }
                    cv::ExifReader reader;
                    reader.parseExif(buffer.data(), buffer.size());
                    const auto entry = reader.getTag(cv::ORIENTATION);
                    if (entry.tag != cv::INVALID_TAG) {
                        orientation = FromExifOrientation(static_cast<ExifOrientation>(entry.field_u16));
                    }
                    exif_parsed = true;
                }
                io_stream->seek(io_stream->instance, end_of_chunk_pos, SEEK_SET);
            }
        }
    } else {
        NVIMGCDCS_LOG_ERROR(Logger::get(), "Unexpected chunk type");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    nchannels += static_cast<int>(alpha);

    image_info->sample_format = nchannels >= 3 ? NVIMGCDCS_SAMPLEFORMAT_P_RGB : NVIMGCDCS_SAMPLEFORMAT_P_Y;
    image_info->orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
    image_info->chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;
    image_info->color_spec = NVIMGCDCS_COLORSPEC_SRGB;
    image_info->num_planes = nchannels;
    image_info->orientation = orientation;
    for (size_t p = 0; p < nchannels; p++) {
        image_info->plane_info[p].height = height;
        image_info->plane_info[p].width = width;
        image_info->plane_info[p].num_channels = 1;
        image_info->plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace

WebpParserPlugin::WebpParserPlugin()
    : parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, nullptr,
          this,          // instance
          "webp_parser", // id
          "webp",        // codec_type
          static_can_parse, static_create, Parser::static_destroy, Parser::static_get_image_info}
{
}

nvimgcdcsParserDesc_t* WebpParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t WebpParserPlugin::canParse(bool* result, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t length;
    io_stream->size(io_stream->instance, &length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);

    if (length < (4 + 4 + 4)) { // RIFF + file size + WEBP
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    // https://developers.google.com/speed/webp/docs/riff_container#webp_file_header
    auto riff = ReadValue<chunk_type_t>(io_stream);
    if (riff != RIFF_TAG) {
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }
    io_stream->skip(io_stream->instance, sizeof(uint32_t)); // file_size
    auto webp = ReadValue<chunk_type_t>(io_stream);
    if (webp != WEBP_TAG) {
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    auto chunk_type = ReadValue<chunk_type_t>(io_stream);
    if (chunk_type != VP8_TAG && chunk_type != VP8L_TAG && chunk_type != VP8X_TAG) {
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }
    *result = true;
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t WebpParserPlugin::static_can_parse(void* instance, bool* result, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE(Logger::get(), "webp_parser_can_parse");
        CHECK_NULL(instance);
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
        auto handle = reinterpret_cast<WebpParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(Logger::get(), "Could not check if code stream can be parsed - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

WebpParserPlugin::Parser::Parser()
{
}

nvimgcdcsStatus_t WebpParserPlugin::create(nvimgcdcsParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcdcsParser_t>(new WebpParserPlugin::Parser());
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t WebpParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_LOG_TRACE(Logger::get(), "webp_parser_create");
        CHECK_NULL(instance);
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<WebpParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(Logger::get(), "Could not create webp parser - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t WebpParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        NVIMGCDCS_LOG_TRACE(Logger::get(), "webp_parser_destroy");
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<WebpParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(Logger::get(), "Could not destroy webp parser - " << e.what());
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t WebpParserPlugin::Parser::getImageInfo(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    NVIMGCDCS_LOG_TRACE(Logger::get(), "webp_parser_get_image_info");
    try {
        return GetImageInfoImpl(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(Logger::get(), "Could not retrieve image info from png stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    }
}

nvimgcdcsStatus_t WebpParserPlugin::Parser::static_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE(Logger::get(), "webp_parser_get_image_info");
        CHECK_NULL(parser);
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        auto handle = reinterpret_cast<WebpParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(Logger::get(), "Could not retrieve image info from webp code stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

class WebpParserExtension
{
  public:
    explicit WebpParserExtension(const nvimgcdcsFrameworkDesc_t* framework)
        : framework_(framework)
    {
        framework->registerParser(framework->instance, webp_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
    }
    ~WebpParserExtension() { framework_->unregisterParser(framework_->instance, webp_parser_plugin_.getParserDesc()); }

  private:
    const nvimgcdcsFrameworkDesc_t* framework_;
    WebpParserPlugin webp_parser_plugin_;
};

nvimgcdcsStatus_t webp_parser_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework)
{
    NVIMGCDCS_LOG_TRACE(Logger::get(), "webp_parser_extension_create");
    try {
        CHECK_NULL(framework)
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new WebpParserExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t webp_parser_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE(Logger::get(), "webp_parser_extension_destroy");
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcdcs::WebpParserExtension*>(extension);
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t webp_parser_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "webp_parser_extension",
    NVIMGCDCS_VER,
    NVIMGCDCS_EXT_API_VER,

    webp_parser_extension_create,
    webp_parser_extension_destroy
};
// clang-format on

nvimgcdcsStatus_t get_webp_parser_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = webp_parser_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace nvimgcdcs