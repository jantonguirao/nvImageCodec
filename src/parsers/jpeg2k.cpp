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
#include <string.h>
#include <vector>

#include "exception.h"
#include "log_ext.h"

#include "parsers/byte_io.h"
#include "parsers/exif.h"

// Reference: https://www.itu.int/rec/T-REC-T.800-200208-S

namespace nvimgcdcs {

namespace {

const std::array<uint8_t, 12> JP2_SIGNATURE = {0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20, 0x0d, 0x0a, 0x87, 0x0a};
const std::array<uint8_t, 2> J2K_SIGNATURE = {0xff, 0x4f};

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

bool ReadBoxHeader(block_type_t& block_type, uint32_t& block_size, nvimgcdcsIoStreamDesc_t* io_stream)
{
    block_size = ReadValueBE<uint32_t>(io_stream);
    block_type = ReadValue<block_type_t>(io_stream);
    return true;
}

void SkipBox(nvimgcdcsIoStreamDesc_t* io_stream, block_type_t expected_block, const char* box_description)
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
        sample_type = sign_component ? NVIMGCDCS_SAMPLE_DATA_TYPE_INT16 : NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
    } else if (bits_per_component <= 8) {
        sample_type = sign_component ? NVIMGCDCS_SAMPLE_DATA_TYPE_INT8 : NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
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

JPEG2KParserPlugin::JPEG2KParserPlugin(const nvimgcdcsFrameworkDesc_t* framework)
    : framework_(framework)
    , parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, nullptr, this, plugin_id_, "jpeg2k", static_can_parse, static_create,
          Parser::static_destroy, Parser::static_get_image_info}
{
}

nvimgcdcsParserDesc_t* JPEG2KParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::canParse(int* result, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "jpeg2k_parser_can_parse");
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
    nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t bitstream_size = 0;
    io_stream->size(io_stream->instance, &bitstream_size);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    *result = 0;

    std::array<uint8_t, 12> bitstream_start;
    size_t read_nbytes = 0;
    io_stream->read(io_stream->instance, &read_nbytes, bitstream_start.data(), bitstream_start.size());
    if (read_nbytes < bitstream_start.size())
        return NVIMGCDCS_STATUS_SUCCESS;

    if (!memcmp(bitstream_start.data(), JP2_SIGNATURE.data(), JP2_SIGNATURE.size()))
        *result = 1;
    else if (!memcmp(bitstream_start.data(), J2K_SIGNATURE.data(), J2K_SIGNATURE.size()))
        *result = 1;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not check if code stream can be parsed - " << e.what());
        return NVIMGCDCS_EXTENSION_STATUS_INTERNAL_ERROR;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::static_can_parse(void* instance, int* result, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<JPEG2KParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_EXTENSION_STATUS_INVALID_PARAMETER;
    }
}

JPEG2KParserPlugin::Parser::Parser(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework)
    : plugin_id_(plugin_id)
    , framework_(framework)
{
    NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "jpeg2k_parser_destroy");
}

nvimgcdcsStatus_t JPEG2KParserPlugin::create(nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "jpeg2k_parser_create");
        CHECK_NULL(parser);
        *parser = reinterpret_cast<nvimgcdcsParser_t>(new JPEG2KParserPlugin::Parser(plugin_id_, framework_));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not create jpeg2k parser - " << e.what());
        return NVIMGCDCS_EXTENSION_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<JPEG2KParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_EXTENSION_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<JPEG2KParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_EXTENSION_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::parseJP2(nvimgcdcsIoStreamDesc_t* io_stream)
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
                        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Invalid JPEG2K image header");
                        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
                    }
                    height = ReadValueBE<uint32_t>(io_stream);
                    width = ReadValueBE<uint32_t>(io_stream);
                    num_components = ReadValueBE<uint16_t>(io_stream);

                    if (num_components > NVIMGCDCS_MAX_NUM_PLANES) {
                        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Too many components " << num_components);
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
            return parseCodeStream(io_stream); // parsing ends here
        }
    }
    return NVIMGCDCS_STATUS_BAD_CODESTREAM; //  didn't parse codestream
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::parseCodeStream(nvimgcdcsIoStreamDesc_t* io_stream)
{
    auto marker = ReadValueBE<uint16_t>(io_stream);
    if (marker != SOC_marker) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "SOC marker not found");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }
    // SOC should be followed by SIZ. Figure A.3
    marker = ReadValueBE<uint16_t>(io_stream);
    if (marker != SIZ_marker) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "SIZ marker not found");
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    auto marker_size = ReadValueBE<uint16_t>(io_stream);
    if (marker_size < 41 || marker_size > 49190) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Invalid SIZ marker size");
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
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Too many components " << num_components);
        return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
    }

    for (int i = 0; i < CSiz; i++) {
        Ssiz[i] = ReadValueBE<uint8_t>(io_stream);
        XRSiz[i] = ReadValue<uint8_t>(io_stream);
        YRSiz[i] = ReadValue<uint8_t>(io_stream);
        if (bits_per_component != DIFFERENT_BITDEPTH_PER_COMPONENT && Ssiz[i] != bits_per_component) {
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "SSiz is expected to match BPC from image header box");
            return NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED;
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::getImageInfo(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "jpeg2k_parser_get_image_info");
    try {
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        num_components = 0;
        height = 0xFFFFFFFF, width = 0xFFFFFFFF;
        bits_per_component = DIFFERENT_BITDEPTH_PER_COMPONENT;
        color_spec = NVIMGCDCS_COLORSPEC_UNKNOWN;
        XSiz = 0;
        YSiz = 0;
        XOSiz = 0;
        YOSiz = 0;
        CSiz = 0;

        nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
        size_t bitstream_size = 0;
        io_stream->size(io_stream->instance, &bitstream_size);
        if (bitstream_size < 12) {
            return NVIMGCDCS_STATUS_SUCCESS;
        }
        io_stream->seek(io_stream->instance, 0, SEEK_SET);

        if (image_info->type != NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO) {
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Unexpected structure type");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        strcpy(image_info->codec_name, "jpeg2k");
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
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Unexpected number of components in main header versus image header box");
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
            image_info->plane_info[p].precision =
                ((image_info->plane_info[p].sample_type >> 8) & 0xff) == (Ssiz[p] & 0x7F) + 1 ? 0 : (Ssiz[p] & 0x7F) + 1;
        }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from jpeg2k stream - " << e.what());
        return NVIMGCDCS_EXTENSION_STATUS_INTERNAL_ERROR;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t JPEG2KParserPlugin::Parser::static_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<JPEG2KParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_EXTENSION_STATUS_INVALID_PARAMETER;
    }
}

class Jpeg2kParserExtension
{
  public:
    explicit Jpeg2kParserExtension(const nvimgcdcsFrameworkDesc_t* framework)
        : framework_(framework)
        , jpeg2k_parser_plugin_(framework)
    {
        framework->registerParser(framework->instance, jpeg2k_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
    }
    ~Jpeg2kParserExtension() { framework_->unregisterParser(framework_->instance, jpeg2k_parser_plugin_.getParserDesc()); }

    static nvimgcdcsStatus_t jpeg2k_parser_extension_create(
        void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework)
{
    try {
        CHECK_NULL(framework)
            NVIMGCDCS_LOG_TRACE(framework, "jpeg2k_parser_ext", "jpeg2k_parser_extension_create");
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new Jpeg2kParserExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

    static nvimgcdcsStatus_t jpeg2k_parser_extension_destroy(nvimgcdcsExtension_t extension)
{
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcdcs::Jpeg2kParserExtension*>(extension);
            NVIMGCDCS_LOG_TRACE(ext_handle->framework_, "jpeg2k_parser_ext", "jpeg2k_parser_extension_destroy");
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

  private:
    const nvimgcdcsFrameworkDesc_t* framework_;
    JPEG2KParserPlugin jpeg2k_parser_plugin_;
};

// clang-format off
nvimgcdcsExtensionDesc_t jpeg2k_parser_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "jpeg2k_parser_extension",
    NVIMGCDCS_VER, 
    NVIMGCDCS_EXT_API_VER,

    Jpeg2kParserExtension::jpeg2k_parser_extension_create,
    Jpeg2kParserExtension::jpeg2k_parser_extension_destroy
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