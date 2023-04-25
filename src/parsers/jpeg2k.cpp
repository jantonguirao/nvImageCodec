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

#include "exception.h"
#include "log.h"
#include "logger.h"
#include "parsers/byte_io.h"
#include "parsers/exif.h"

// Reference: https://www.itu.int/rec/T-REC-T.800-200208-S

namespace nvimgcdcs {

namespace {

const std::array<uint8_t, 12> JP2_SIGNATURE = {0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20, 0x0d, 0x0a, 0x87, 0x0a};
const std::array<uint8_t, 2>  J2K_SIGNATURE = {0xff, 0x4f};

using block_type_t = std::array<uint8_t, 4>;
const block_type_t jp2_signature = {'j', 'P', ' ', ' '};    // JPEG2000 signature
const block_type_t jp2_file_type = {'f', 't', 'y', 'p'};    // File type
const block_type_t jp2_header = {'j', 'p', '2', 'h'};       // JPEG2000 header (super box)
const block_type_t jp2_image_header = {'i', 'h', 'd', 'r'}; // Image header
const block_type_t jp2_colour_spec = {'c', 'o', 'l', 'r'};  // Color specification
const block_type_t jp2_code_stream = {'j', 'p', '2', 'c'};  // Contiguous code stream
const block_type_t jp2_url = {'u', 'r', 'l', ' '};          // Data entry URL
const block_type_t jp2_palette = {'p', 'c', 'l', 'r'};      // Palette
const block_type_t jp2_cmap = {'c', 'm', 'a', 'p'};         // Component mapping
const block_type_t jp2_cdef = {'c', 'd', 'e', 'f'};         // Channel definition
const block_type_t jp2_dtbl = {'d', 't', 'b', 'l'};         // Data reference
const block_type_t jp2_bpcc = {'b', 'p', 'c', 'c'};         // Bits per component
const block_type_t jp2_jp2 = {'j', 'p', '2', ' '};          // File type fields

enum jpeg2k_marker_t : uint16_t
{
    SOC_marker = 0xFF4F,
    SIZ_marker = 0xFF51
};

const uint8_t DIFFERENT_BITDEPTH_PER_COMPONENT = 0xFF;

bool ReadBoxHeader(block_type_t& block_type, uint32_t& block_size, nvimgcdcsIoStreamDesc_t io_stream)
{
    block_size = ReadValueBE<uint32_t>(io_stream);
    block_type = ReadValue<block_type_t>(io_stream);
    return true;
}

void SkipBox(nvimgcdcsIoStreamDesc_t io_stream, block_type_t expected_block, const char* box_description)
{
    auto block_size = ReadValueBE<uint32_t>(io_stream);
    auto block_type = ReadValue<block_type_t>(io_stream);
    if (block_type != expected_block)
        throw std::runtime_error(std::string("Failed to read ") + std::string(box_description));
    io_stream->skip(io_stream->instance, block_size - sizeof(block_size) - sizeof(block_type));
}

template <typename T, typename V>
constexpr inline T DivUp(T x, V d)
{
    return (x + d - 1) / d;
}

nvimgcdcsSampleDataType_t BitsPerComponentToType(uint8_t bits_per_component)
{
    auto sign_component = bits_per_component >> 7;
    bits_per_component = bits_per_component & 0x7f;
    bits_per_component += 1;
    auto sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UNSUPPORTED;
    if (bits_per_component <= 16 && bits_per_component > 8) {
        sample_type = sign_component ? NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16 : NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
    } else if (bits_per_component <= 8) {
        sample_type = sign_component ? NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8 : NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
    }
    return sample_type;
}

nvimgcdcsChromaSubsampling_t XRSizYRSizToSubsampling(uint8_t CSiz, const uint8_t* XRSiz, const uint8_t* YRSiz)
{
    if (CSiz == 3 || CSiz == 4) {
        if ((XRSiz[0] == 1) && (XRSiz[1] == 2) && (XRSiz[2] == 2) && (YRSiz[0] == 1) && (YRSiz[1] == 2) && (YRSiz[2] == 2)) {
            return NVIMGCDCS_SAMPLING_420;
        } else if ((XRSiz[0] == 1) && (XRSiz[1] == 2) && (XRSiz[2] == 2) && (YRSiz[0] == 1) && (YRSiz[1] == 1) && (YRSiz[2] == 1)) {
            return NVIMGCDCS_SAMPLING_422;
        } else if ((XRSiz[0] == 1) && (XRSiz[1] == 1) && (XRSiz[2] == 1) && (YRSiz[0] == 1) && (YRSiz[1] == 1) && (YRSiz[2] == 1)) {
            return NVIMGCDCS_SAMPLING_444;
        } else {
            return NVIMGCDCS_SAMPLING_UNSUPPORTED;
        }
    } else {
        for (uint8_t i = 0; i < CSiz; i++) {
            if ((XRSiz[0] != 1) || (XRSiz[1] != 1) || (XRSiz[2] != 1) || (YRSiz[0] != 1) || (YRSiz[1] != 1) || (YRSiz[2] != 1))
                return NVIMGCDCS_SAMPLING_UNSUPPORTED;
        }
        return NVIMGCDCS_SAMPLING_NONE;
    }
}

} // namespace

JPEG2KParserPlugin::JPEG2KParserPlugin()
    : parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, nullptr,
          this,            // instance
          "jpeg2k_parser", // id
          0x00000100,      // version
          "jpeg2k",        // codec_type
          static_can_parse, static_create, Parser::static_destroy, Parser::static_create_parse_state, Parser::static_destroy_parse_state,
          Parser::static_get_image_info, Parser::static_get_capabilities}
{}

struct nvimgcdcsParserDesc* JPEG2KParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::canParse(bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    size_t bitstream_size = 0;
    io_stream->size(io_stream->instance, &bitstream_size);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    *result = false;
    
    std::array<uint8_t, 12> bitstream_start;
    size_t read_nbytes = 0;
    io_stream->read(io_stream->instance, &read_nbytes, bitstream_start.data(), bitstream_start.size());
    if (read_nbytes < bitstream_start.size())
        return NVIMGCDCS_STATUS_SUCCESS;

    if(!memcmp(bitstream_start.data(), JP2_SIGNATURE.data(), JP2_SIGNATURE.size()))
        *result = true;
    else if(!memcmp(bitstream_start.data(), J2K_SIGNATURE.data(), J2K_SIGNATURE.size()))
        *result = true;
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::static_can_parse(void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
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
{}

nvimgcdcsStatus_t JPEG2KParserPlugin::create(nvimgcdcsParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcdcsParser_t>(new JPEG2KParserPlugin::Parser());
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

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_get_capabilities(
    nvimgcdcsParser_t parser, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_LOG_TRACE("jpeg2k_get_capabilities");
        CHECK_NULL(parser);
        CHECK_NULL(capabilities);
        CHECK_NULL(size);
        auto handle = reinterpret_cast<JPEG2KParserPlugin::Parser*>(parser);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve jpeg parser capabilites - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::createParseState(nvimgcdcsParseState_t* parse_state)
{
    // TODO(janton): remove this API
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_create_parse_state(nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state)
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

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_destroy_parse_state(nvimgcdcsParseState_t parse_state)
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

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::parseJP2(nvimgcdcsIoStreamDesc_t io_stream)
{
    uint32_t block_size;
    block_type_t block_type;
    SkipBox(io_stream, jp2_signature, "JPEG2K signature");
    SkipBox(io_stream, jp2_file_type, "JPEG2K file type");
    while (ReadBoxHeader(block_type, block_size, io_stream)) {
        if (block_type == jp2_header) { // superbox
            auto remaining_bytes = block_size - sizeof(block_size) - sizeof(block_type);
            while (remaining_bytes > 0) {
                ReadBoxHeader(block_type, block_size, io_stream);
                if (block_type == jp2_image_header) { // Ref. I.5.3.1 Image Header box
                    if (block_size != 22) {
                        NVIMGCDCS_LOG_ERROR("Invalid JPEG2K image header");
                        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
                    }
                    height = ReadValueBE<uint32_t>(io_stream);
                    width = ReadValueBE<uint32_t>(io_stream);
                    num_components = ReadValueBE<uint16_t>(io_stream);

                    if (num_components > NVIMGCDCS_MAX_NUM_PLANES) {
                        NVIMGCDCS_LOG_ERROR("Too many components " << num_components);
                        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
                    }

                    bits_per_component = ReadValueBE<uint8_t>(io_stream);
                    io_stream->skip(io_stream->instance, sizeof(uint8_t)); // compression_type
                    io_stream->skip(io_stream->instance, sizeof(uint8_t)); // color_space_unknown
                    io_stream->skip(io_stream->instance, sizeof(uint8_t)); // IPR
                } else if (block_type == jp2_colour_spec && color_spec == NVIMGCDCS_COLORSPEC_UNKNOWN) {
                    auto method = ReadValueBE<uint8_t>(io_stream);
                    io_stream->skip(io_stream->instance, sizeof(int8_t)); // precedence
                    io_stream->skip(io_stream->instance, sizeof(int8_t)); // colourspace approximation
                    auto enumCS = ReadValueBE<uint32_t>(io_stream);
                    if (method == 1) {
                        switch (enumCS) {
                        case 16: // sRGB
                            color_spec = NVIMGCDCS_COLORSPEC_SRGB;
                            break;
                        case 17: // Greyscale
                            color_spec = NVIMGCDCS_COLORSPEC_GRAY;
                            break;
                        case 18: // sYCC
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
            return parseCodeStream(io_stream);  // parsing ends here
        }
    }
    return NVIMGCDCS_STATUS_BAD_CODESTREAM;  //  didn't parse codestream
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::parseCodeStream(nvimgcdcsIoStreamDesc_t io_stream)
{
    auto marker = ReadValueBE<uint16_t>(io_stream);
    if (marker != SOC_marker) {
        NVIMGCDCS_LOG_ERROR("SOC marker not found");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }
    // SOC should be followed by SIZ. Figure A.3
    marker = ReadValueBE<uint16_t>(io_stream);
    if (marker != SIZ_marker) {
        NVIMGCDCS_LOG_ERROR("SIZ marker not found");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    auto marker_size = ReadValueBE<uint16_t>(io_stream);
    if (marker_size < 41 || marker_size > 49190) {
        NVIMGCDCS_LOG_ERROR("Invalid SIZ marker size");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    io_stream->skip(io_stream->instance, sizeof(uint16_t)); // RSiz
    XSiz = ReadValueBE<uint32_t>(io_stream);
    YSiz = ReadValueBE<uint32_t>(io_stream);
    XOSiz = ReadValueBE<uint32_t>(io_stream);
    YOSiz = ReadValueBE<uint32_t>(io_stream);
    io_stream->skip(io_stream->instance, sizeof(uint32_t)); // XTSiz
    io_stream->skip(io_stream->instance, sizeof(uint32_t)); // YTSiz
    io_stream->skip(io_stream->instance, sizeof(uint32_t)); // XTOSiz
    io_stream->skip(io_stream->instance, sizeof(uint32_t)); // YTOSiz
    CSiz = ReadValueBE<uint16_t>(io_stream);

    // CSiz in table A.9, minimum of 1 and Max of 16384
    if (CSiz > NVIMGCDCS_MAX_NUM_PLANES) {
        NVIMGCDCS_LOG_ERROR("Too many components " << num_components);
        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
    }

    for (int i = 0; i < CSiz; i++) {
        Ssiz[i] = ReadValueBE<uint8_t>(io_stream);
        XRSiz[i] = ReadValue<uint8_t>(io_stream);
        YRSiz[i] = ReadValue<uint8_t>(io_stream);
        if (bits_per_component != DIFFERENT_BITDEPTH_PER_COMPONENT && Ssiz[i] != bits_per_component) {
            NVIMGCDCS_LOG_ERROR("SSiz is expected to match BPC from image header box");
            return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::getImageInfo(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    NVIMGCDCS_LOG_TRACE("jpeg2k_parser_get_image_info");
    try {
        num_components = 0;
        height = 0xFFFFFFFF, width = 0xFFFFFFFF;
        bits_per_component = DIFFERENT_BITDEPTH_PER_COMPONENT;
        color_spec = NVIMGCDCS_COLORSPEC_UNKNOWN;
        XSiz = 0;
        YSiz = 0;
        XOSiz = 0;
        YOSiz = 0;
        CSiz = 0;

        nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
        size_t bitstream_size = 0;
        io_stream->size(io_stream->instance, &bitstream_size);
        if (bitstream_size < 12) {
            return NVIMGCDCS_STATUS_SUCCESS;
        }
        io_stream->seek(io_stream->instance, 0, SEEK_SET);

        if (image_info->type != NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO) {
            NVIMGCDCS_LOG_ERROR("Unexpected structure type");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }

        std::array<uint8_t, 12> bitstream_start;
        size_t read_nbytes = 0;
        io_stream->read(io_stream->instance, &read_nbytes, bitstream_start.data(), bitstream_start.size());
        io_stream->seek(io_stream->instance, 0, SEEK_SET);
        if (read_nbytes < bitstream_start.size())
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;

        nvimgcdcsStatus_t status = NVIMGCDCS_STATUS_BAD_CODESTREAM;
        if (!memcmp(bitstream_start.data(), JP2_SIGNATURE.data(), JP2_SIGNATURE.size()))
            status = parseJP2(io_stream);
        else if (!memcmp(bitstream_start.data(), J2K_SIGNATURE.data(), J2K_SIGNATURE.size()))
            status = parseCodeStream(io_stream);

        if (status != NVIMGCDCS_STATUS_SUCCESS)
            return status;

        num_components = num_components > 0 ? num_components : CSiz;
        if (CSiz != num_components) {
            NVIMGCDCS_LOG_ERROR("Unexpected number of components in main header versus image header box");
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }

        image_info->sample_format = num_components > 1 ? NVIMGCDCS_SAMPLEFORMAT_P_RGB : NVIMGCDCS_SAMPLEFORMAT_P_Y;
        image_info->orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
        image_info->chroma_subsampling = XRSizYRSizToSubsampling(CSiz, &XRSiz[0], &YRSiz[0]);
        image_info->color_spec = color_spec;
        image_info->num_planes = num_components;
        for (int p = 0; p < num_components; p++) {
            image_info->plane_info[p].height = DivUp(YSiz - YOSiz, YRSiz[p]);
            image_info->plane_info[p].width = DivUp(XSiz - XOSiz, XRSiz[p]);
            image_info->plane_info[p].num_channels = 1;
            image_info->plane_info[p].sample_type = BitsPerComponentToType(Ssiz[p]);
        }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR("Could not retrieve image info from jpeg stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
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

nvimgcdcsStatus_t jpeg2k_parser_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    NVIMGCDCS_LOG_TRACE("extension_create");

    framework->registerParser(framework->instance, jpeg2k_parser_plugin.getParserDesc());

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t jpeg2k_parser_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("jpeg2k_parser_extension_destroy");

    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t jpeg2k_parser_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

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

} // namespace nvimgcdcs