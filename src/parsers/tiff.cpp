/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "parsers/tiff.h"
#include <nvimgcodecs.h>
#include <string.h>
#include <vector>

#include "exception.h"
#include "exif_orientation.h"
#include "log.h"
#include "logger.h"
#include "parsers/byte_io.h"
#include "parsers/exif.h"

namespace nvimgcdcs {

namespace {

constexpr int ENTRY_SIZE = 12;

enum TiffTag : uint16_t
{
    WIDTH_TAG = 256,
    HEIGHT_TAG = 257,
    PHOTOMETRIC_INTERPRETATION_TAG = 262,
    ORIENTATION_TAG = 274,
    SAMPLESPERPIXEL_TAG = 277
};

enum TiffDataType : uint16_t
{
    TYPE_WORD = 3,
    TYPE_DWORD = 4
};

constexpr int PHOTOMETRIC_PALETTE = 3;

using tiff_magic_t = std::array<uint8_t, 4>;
constexpr tiff_magic_t le_header = {'I', 'I', 42, 0}, be_header = {'M', 'M', 0, 42};

template <typename T, bool is_little_endian>
T TiffRead(nvimgcdcsIoStreamDesc_t* io_stream)
{
    if constexpr (is_little_endian) {
        return ReadValueLE<T>(io_stream);
    } else {
        return ReadValueBE<T>(io_stream);
    }
}

template <bool is_little_endian>
nvimgcdcsStatus_t GetInfoImpl(nvimgcdcsImageInfo_t* info, nvimgcdcsIoStreamDesc_t* io_stream)
{
    io_stream->seek(io_stream->instance, 4, SEEK_SET);
    const auto ifd_offset = TiffRead<uint32_t, is_little_endian>(io_stream);
    io_stream->seek(io_stream->instance, ifd_offset, SEEK_SET);
    const auto entry_count = TiffRead<uint16_t, is_little_endian>(io_stream);

    strcpy(info->codec_name, "tiff");
    info->color_spec = NVIMGCDCS_COLORSPEC_UNKNOWN;
    info->chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;
    info->sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
    info->orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};

    bool width_read = false, height_read = false, samples_per_px_read = false, palette_read = false;
    int64_t width = 0, height = 0, nchannels = 0;

    for (int entry_idx = 0; entry_idx < entry_count; entry_idx++) {
        const auto entry_offset = ifd_offset + sizeof(uint16_t) + entry_idx * ENTRY_SIZE;
        io_stream->seek(io_stream->instance, entry_offset, SEEK_SET);
        const auto tag_id = TiffRead<uint16_t, is_little_endian>(io_stream);
        if (tag_id == WIDTH_TAG || tag_id == HEIGHT_TAG || tag_id == SAMPLESPERPIXEL_TAG || tag_id == ORIENTATION_TAG ||
            tag_id == PHOTOMETRIC_INTERPRETATION_TAG) {
            const auto value_type = TiffRead<uint16_t, is_little_endian>(io_stream);
            const auto value_count = TiffRead<uint32_t, is_little_endian>(io_stream);
            if (value_count != 1) {
                NVIMGCDCS_LOG_ERROR("Unexpected value count");
                return NVIMGCDCS_STATUS_BAD_CODESTREAM;
            }

            int64_t value;
            if (value_type == TYPE_WORD) {
                value = TiffRead<uint16_t, is_little_endian>(io_stream);
            } else if (value_type == TYPE_DWORD) {
                value = TiffRead<uint32_t, is_little_endian>(io_stream);
            } else {
                NVIMGCDCS_LOG_ERROR("Couldn't read TIFF image dims");
                return NVIMGCDCS_STATUS_BAD_CODESTREAM;
            }

            if (tag_id == WIDTH_TAG) {
                width = value;
                width_read = true;
            } else if (tag_id == HEIGHT_TAG) {
                height = value;
                height_read = true;
            } else if (tag_id == ORIENTATION_TAG) {
                info->orientation = FromExifOrientation(static_cast<ExifOrientation>(value));
            } else if (tag_id == SAMPLESPERPIXEL_TAG && !palette_read) {
                // If the palette is present, the SAMPLESPERPIXEL tag is always set to 1, so it does not
                // indicate the actual number of channels. That's why we ignore it for palette images.
                nchannels = value;
                samples_per_px_read = true;
            } else if (tag_id == PHOTOMETRIC_INTERPRETATION_TAG && value == PHOTOMETRIC_PALETTE) {
                nchannels = 3;
                palette_read = true;
            }
        }
        if (width_read && height_read && palette_read)
            break;
    }

    if (!width_read || !height_read || (!samples_per_px_read && !palette_read)) {
        NVIMGCDCS_LOG_ERROR("Couldn't read TIFF image dims");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    info->num_planes = nchannels;
    for (size_t p = 0; p < info->num_planes; p++) {
        info->plane_info[p].height = height;
        info->plane_info[p].width = width;
        info->plane_info[p].num_channels = 1;
        info->plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
    }
    if (nchannels == 1)
        info->sample_format = NVIMGCDCS_SAMPLEFORMAT_P_Y;
    else
        info->sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace

TIFFParserPlugin::TIFFParserPlugin()
    : parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, nullptr,
          this,          // instance
          "tiff_parser", // id
          "tiff",        // codec_type
          static_can_parse, static_create, Parser::static_destroy, Parser::static_get_image_info}
{
}

struct nvimgcdcsParserDesc* TIFFParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t TIFFParserPlugin::canParse(bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t length;
    io_stream->size(io_stream->instance, &length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    if (length < 4) {
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }
    tiff_magic_t header = ReadValue<tiff_magic_t>(io_stream);
    *result = header == le_header || header == be_header;
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t TIFFParserPlugin::static_can_parse(void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("tiff_parser_can_parse");
        CHECK_NULL(instance);
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
        auto handle = reinterpret_cast<TIFFParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not check if code stream can be parsed - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

TIFFParserPlugin::Parser::Parser()
{
}

nvimgcdcsStatus_t TIFFParserPlugin::create(nvimgcdcsParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcdcsParser_t>(new TIFFParserPlugin::Parser());
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t TIFFParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("tiff_parser_create");
        CHECK_NULL(instance);
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<TIFFParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not create tiff parser - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t TIFFParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        NVIMGCDCS_LOG_TRACE("tiff_parser_destroy");
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<TIFFParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not destroy tiff parser - " << e.what());
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t TIFFParserPlugin::Parser::getImageInfo(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    NVIMGCDCS_LOG_TRACE("tiff_parser_get_image_info");
    try {
        nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
        size_t length;
        io_stream->size(io_stream->instance, &length);
        io_stream->seek(io_stream->instance, 0, SEEK_SET);

        tiff_magic_t header = ReadValue<tiff_magic_t>(io_stream);
        nvimgcdcsStatus_t ret = NVIMGCDCS_STATUS_SUCCESS;
        if (header == le_header) {
            ret = GetInfoImpl<true>(image_info, io_stream);
        } else if (header == be_header) {
            ret = GetInfoImpl<false>(image_info, io_stream);
        } else {
            // should not happen (because canParse returned result==true)
            NVIMGCDCS_LOG_ERROR("Logic error");
            ret = NVIMGCDCS_STATUS_INTERNAL_ERROR;
        }
        return ret;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from tiff stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t TIFFParserPlugin::Parser::static_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE("tiff_parser_get_image_info");
        CHECK_NULL(parser);
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        auto handle = reinterpret_cast<TIFFParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from tiff code stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

class TiffParserExtension
{
  public:
    explicit TiffParserExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
    {
        framework->registerParser(framework->instance, tiff_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
    }
    ~TiffParserExtension() { framework_->unregisterParser(framework_->instance, tiff_parser_plugin_.getParserDesc()); }

  private:
    const nvimgcdcsFrameworkDesc_t framework_;
    TIFFParserPlugin tiff_parser_plugin_;
};

nvimgcdcsStatus_t tiff_parser_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    NVIMGCDCS_LOG_TRACE("tiff_parser_extension_create");
    try {
        CHECK_NULL(framework)
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new TiffParserExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t tiff_parser_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("tiff_parser_extension_destroy");
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcdcs::TiffParserExtension*>(extension);
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t tiff_parser_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,
   
    NULL,
    "tiff_parser_extension",
    NVIMGCDCS_VER,
    NVIMGCDCS_EXT_API_VER,

    tiff_parser_extension_create,
    tiff_parser_extension_destroy
};
// clang-format on

nvimgcdcsStatus_t get_tiff_parser_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = tiff_parser_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace nvimgcdcs