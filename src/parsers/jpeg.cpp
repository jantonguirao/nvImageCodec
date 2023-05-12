/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "parsers/jpeg.h"
#include <nvimgcodecs.h>
#include <vector>

#include "exception.h"
#include "exif_orientation.h"
#include "log.h"
#include "logger.h"
#include "parsers/byte_io.h"
#include "parsers/exif.h"

namespace nvimgcdcs {

using jpeg_marker_t = std::array<uint8_t, 2>;
using jpeg_exif_header_t = std::array<uint8_t, 6>;

namespace {

constexpr jpeg_marker_t sos_marker = {0xff, 0xda};
constexpr jpeg_marker_t soi_marker = {0xff, 0xd8};
constexpr jpeg_marker_t eoi_marker = {0xff, 0xd9};
constexpr jpeg_marker_t app1_marker = {0xff, 0xe1};
constexpr jpeg_marker_t app14_marker = {0xff, 0xee};

constexpr jpeg_exif_header_t exif_header = {'E', 'x', 'i', 'f', 0, 0};

bool IsValidMarker(const jpeg_marker_t& marker)
{
    return marker[0] == 0xff && marker[1] != 0x00;
}

bool IsSofMarker(const jpeg_marker_t& marker)
{
    // According to https://www.w3.org/Graphics/JPEG/itu-t81.pdf table B.1 Marker code assignments
    // SOF markers are from range 0xFFC0-0xFFCF, excluding 0xFFC4, 0xFFC8 and 0xFFCC.
    if (!IsValidMarker(marker) || marker[1] < 0xc0 || marker[1] > 0xcf)
        return false;
    return marker[1] != 0xc4 && marker[1] != 0xc8 && marker[1] != 0xcc;
}

nvimgcdcsSampleDataType_t precision_to_sample_type(int precision)
{
    return precision == 8 ? NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 : NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED;
}

nvimgcdcsChromaSubsampling_t chroma_subsampling_from_factors(
    int ncomponents, uint8_t yh, uint8_t yv, uint8_t uh, uint8_t uv, uint8_t vh, uint8_t vv)
{
    if (ncomponents == 1)
        return NVIMGCDCS_SAMPLING_GRAY;

    if (ncomponents == 3) {
        uint8_t minh = std::min(yh, std::min(uh, vh));
        uint8_t minv = std::min(yv, std::min(uv, vv));

        if (minh == 0 || minv == 0)
            return NVIMGCDCS_SAMPLING_UNSUPPORTED;

        if (yh % minh || uh % minh || vh % minh || yv % minv || uv % minv || vv % minv)
            return NVIMGCDCS_SAMPLING_UNSUPPORTED; // non-integer factors
        yh /= minh;
        uh /= minh;
        vh /= minh;
        yv /= minv;
        uv /= minv;
        vv /= minv;

        if (uh != vh || uv != vv)
            return NVIMGCDCS_SAMPLING_UNSUPPORTED; // in chroma subsamplings we support chroma should have same factors

        if (uh != 1 || uv != 1)
            return NVIMGCDCS_SAMPLING_UNSUPPORTED; // U/V should be 1x1

        if (yh == 1 && yv == 1)
            return NVIMGCDCS_SAMPLING_444;
        else if (yh == 2 && yv == 1)
            return NVIMGCDCS_SAMPLING_422;
        else if (yh == 2 && yv == 2)
            return NVIMGCDCS_SAMPLING_420;
        else if (yh == 1 && yv == 2)
            return NVIMGCDCS_SAMPLING_440;
        else if (yh == 4 && yv == 1)
            return NVIMGCDCS_SAMPLING_411;
        else if (yh == 4 && yv == 2)
            return NVIMGCDCS_SAMPLING_410;
        else if (yh == 2 && yv == 4)
            return NVIMGCDCS_SAMPLING_410V;
    }
    return NVIMGCDCS_SAMPLING_UNSUPPORTED;
}

} // namespace

JPEGParserPlugin::JPEGParserPlugin()
    : parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, nullptr,
          this,          // instance
          "jpeg_parser", // id
          0x00000100,    // version
          "jpeg",        // codec_type
          static_can_parse, static_create, Parser::static_destroy, Parser::static_create_parse_state, Parser::static_destroy_parse_state,
          Parser::static_get_image_info, Parser::static_get_capabilities}
{
}

struct nvimgcdcsParserDesc* JPEGParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t JPEGParserPlugin::canParse(bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    auto signature = ReadValue<jpeg_marker_t>(io_stream);
    *result = (signature == soi_marker);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEGParserPlugin::static_can_parse(void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg_parser_can_parse");
        CHECK_NULL(instance);
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
        auto handle = reinterpret_cast<JPEGParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not check if code stream can be parsed - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

JPEGParserPlugin::Parser::Parser()
{
}

nvimgcdcsStatus_t JPEGParserPlugin::create(nvimgcdcsParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcdcsParser_t>(new JPEGParserPlugin::Parser());
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEGParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg_parser_create");
        CHECK_NULL(instance);
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<JPEGParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not create jpeg parser - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEGParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg_parser_destroy");
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<JPEGParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not destroy jpeg parser - " << e.what());
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEGParserPlugin::Parser::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t JPEGParserPlugin::Parser::static_get_capabilities(
    nvimgcdcsParser_t parser, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg_get_capabilities");
        CHECK_NULL(parser);
        CHECK_NULL(capabilities);
        CHECK_NULL(size);
        auto handle = reinterpret_cast<JPEGParserPlugin::Parser*>(parser);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve jpeg parser capabilites - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t JPEGParserPlugin::Parser::createParseState(nvimgcdcsParseState_t* parse_state)
{
    // TODO(janton): remove this API
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEGParserPlugin::Parser::static_create_parse_state(nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state)
{
    try {
        NVIMGCDCS_LOG_TRACE("JPEG create_parse_state");
        CHECK_NULL(parser);
        CHECK_NULL(parse_state);
        auto handle = reinterpret_cast<JPEGParserPlugin::Parser*>(parser);
        return handle->createParseState(parse_state);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not create jpeg parse state - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t JPEGParserPlugin::Parser::static_destroy_parse_state(nvimgcdcsParseState_t parse_state)
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

nvimgcdcsStatus_t JPEGParserPlugin::Parser::getImageInfo(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    NVIMGCDCS_LOG_TRACE("jpeg_parser_get_image_info");
    try {
        size_t size = 0;
        nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
        io_stream->size(io_stream->instance, &size);
        io_stream->seek(io_stream->instance, 0, SEEK_SET);

        std::array<uint8_t, 2> signature;
        size_t read_nbytes = 0;
        io_stream->read(io_stream->instance, &read_nbytes, &signature[0], signature.size());
        if (read_nbytes != signature.size()) {
            NVIMGCDCS_LOG_ERROR("Unexpected end-of-stream");
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }
        if (signature != soi_marker) {
            NVIMGCDCS_LOG_ERROR("Unexpected signature");
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }

        bool read_shape = false, read_orientation = false, read_app14 = false;
        uint16_t height = 0, width = 0;
        uint8_t num_components;
        uint8_t precision = 8;
        nvimgcdcsOrientation_t orientation{NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
        int adobe_transform = -1;
        nvimgcdcsChromaSubsampling_t subsampling = NVIMGCDCS_SAMPLING_NONE;
        jpeg_marker_t sof_marker = {};
        while (!read_shape || !read_orientation || !read_app14) {
            jpeg_marker_t marker;
            marker[0] = ReadValue<uint8_t>(io_stream);
            // https://www.w3.org/Graphics/JPEG/itu-t81.pdf section B.1.1.2 Markers
            // Any marker may optionally be preceded by any number of fill bytes,
            // which are bytes assigned code '\xFF'
            do {
                marker[1] = ReadValue<uint8_t>(io_stream);
            } while (marker[1] == 0xff);

            if (!IsValidMarker(marker)) {
                NVIMGCDCS_LOG_ERROR("Invalid marker");
                return NVIMGCDCS_STATUS_BAD_CODESTREAM;
            }
            if (marker == sos_marker)
                break;

            uint16_t size = ReadValueBE<uint16_t>(io_stream);
            size_t offset = 0;
            auto res = io_stream->tell(io_stream->instance, &offset);
            if (res != NVIMGCDCS_STATUS_SUCCESS)
                return res;
            ptrdiff_t next_marker_offset = offset - 2 + size;
            if (IsSofMarker(marker)) {
                sof_marker = marker;
                precision = ReadValue<uint8_t>(io_stream);
                height = ReadValueBE<uint16_t>(io_stream);
                width = ReadValueBE<uint16_t>(io_stream);
                num_components = ReadValue<uint8_t>(io_stream);

                if (num_components > 4)
                    return NVIMGCDCS_STATUS_BAD_CODESTREAM; // should not happen
                std::array<std::pair<uint8_t, uint8_t>, 4> sampling_factors;
                for (int c = 0; c < num_components; c++) {
                    io_stream->skip(io_stream->instance, 1); // component_id
                    auto temp = ReadValue<uint8_t>(io_stream);
                    auto horizontal_sampling_factor = temp >> 4;
                    auto vertical_sampling_factor = temp & 0x0F;
                    sampling_factors[c] = {horizontal_sampling_factor, vertical_sampling_factor};
                    io_stream->skip(io_stream->instance, 1); // quantization table selector
                }
                uint8_t yh = num_components > 0 ? sampling_factors[0].first : 0;
                uint8_t yv = num_components > 0 ? sampling_factors[0].second : 0;
                uint8_t uh = num_components > 1 ? sampling_factors[1].first : 0;
                uint8_t uv = num_components > 1 ? sampling_factors[1].second : 0;
                uint8_t vh = num_components > 2 ? sampling_factors[2].first : 0;
                uint8_t vv = num_components > 2 ? sampling_factors[2].second : 0;
                subsampling = chroma_subsampling_from_factors(num_components, yh, yv, uh, uv, vh, vv);

                read_shape = true;
            } else if (marker == app1_marker && ReadValue<jpeg_exif_header_t>(io_stream) == exif_header) {
                std::vector<uint8_t> exif_block(size - 8);
                io_stream->read(io_stream->instance, &read_nbytes, exif_block.data(), exif_block.size());
                if (read_nbytes != exif_block.size()) {
                    NVIMGCDCS_LOG_ERROR("Failed to read EXIF block");
                    return NVIMGCDCS_STATUS_BAD_CODESTREAM;
                }

                cv::ExifReader reader;
                if (!reader.parseExif(exif_block.data(), exif_block.size()))
                    continue;
                auto entry = reader.getTag(cv::ORIENTATION);
                if (entry.tag != cv::INVALID_TAG) {
                    orientation = FromExifOrientation(static_cast<ExifOrientation>(entry.field_u16));
                    read_orientation = true;
                }
            } else if (marker == app14_marker) {
                constexpr uint16_t app14_data_len = 14;
                constexpr std::array<uint8_t, 5> adobe_signature = {0x41, 0x64, 0x6F, 0x62, 0x65}; /// Adobe in ASCII
                auto signature = ReadValue<std::array<uint8_t, 5>>(io_stream);
                if (size == app14_data_len && signature == adobe_signature) {
                    io_stream->skip(io_stream->instance, 2 + 2 + 2); ////version, flags0, flags1
                    adobe_transform = ReadValue<uint8_t>(io_stream);
                }
            }
            io_stream->seek(io_stream->instance, next_marker_offset, SEEK_SET);
        }
        if (!read_shape) {
            NVIMGCDCS_LOG_ERROR("Failed to read image dimensions");
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }

        image_info->type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
        image_info->sample_format = num_components > 1 ? NVIMGCDCS_SAMPLEFORMAT_P_RGB : NVIMGCDCS_SAMPLEFORMAT_P_Y;
        image_info->orientation = orientation;
        image_info->chroma_subsampling = subsampling;

        switch (num_components) {
        case 1:
            image_info->color_spec = NVIMGCDCS_COLORSPEC_GRAY;
            break;
        case 4:
            image_info->color_spec = adobe_transform == 2 ? NVIMGCDCS_COLORSPEC_YCCK : NVIMGCDCS_COLORSPEC_CMYK;
            break;
        case 3:
            // assume that 3 channels is always going to be YCbCr
            image_info->color_spec = NVIMGCDCS_COLORSPEC_SYCC;
            break;
        default:
            NVIMGCDCS_LOG_ERROR("Unexpected number of channels" << num_components);
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }

        image_info->num_planes = num_components;
        auto sample_type = precision_to_sample_type(precision);
        for (int p = 0; p < num_components; p++) {
            image_info->plane_info[p].width = width;
            image_info->plane_info[p].height = height;
            image_info->plane_info[p].num_channels = 1;
            image_info->plane_info[p].sample_type = sample_type;
        }

        nvimgcdcsJpegImageInfo_t* jpeg_image_info = reinterpret_cast<nvimgcdcsJpegImageInfo_t*>(image_info->next);
        while (jpeg_image_info && jpeg_image_info->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
            jpeg_image_info = reinterpret_cast<nvimgcdcsJpegImageInfo_t*>(jpeg_image_info->next);
        if (jpeg_image_info && jpeg_image_info->type == NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO) {
            jpeg_image_info->encoding = NVIMGCDCS_JPEG_ENCODING_UNKNOWN;
            if (sof_marker[1] >= 0xc0 && sof_marker[1] <= 0xcf)
                jpeg_image_info->encoding = static_cast<nvimgcdcsJpegEncoding_t>(sof_marker[1]);
        }

    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from jpeg stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEGParserPlugin::Parser::static_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg_parser_get_image_info");
        CHECK_NULL(parser);
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        auto handle = reinterpret_cast<JPEGParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from jpeg code stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

static auto jpeg_parser_plugin = JPEGParserPlugin();

nvimgcdcsStatus_t jpeg_parser_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    NVIMGCDCS_LOG_TRACE("extension_create");

    framework->registerParser(framework->instance, jpeg_parser_plugin.getParserDesc());

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t jpeg_parser_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("jpeg_parser_extension_destroy");

    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t jpeg_parser_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "jpeg_parser_extension",  // id
     0x00000100,              // version

    jpeg_parser_extension_create,
    jpeg_parser_extension_destroy
};
// clang-format on

nvimgcdcsStatus_t get_jpeg_parser_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = jpeg_parser_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace nvimgcdcs