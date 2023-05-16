#include "parser.h"

#include <nvimgcodecs.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include <nvjpeg2k.h>
#include "error_handling.h"
#include "log.h"

namespace nvjpeg2k {

NvJpeg2kParserPlugin::NvJpeg2kParserPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, NULL,
          this,              // instance
          "nvjpeg2k_parser", // id
          0x00000100,        // version
          "jpeg2k",          // codec_type
          static_can_parse, static_create, Parser::static_destroy, Parser::static_create_parse_state, Parser::static_destroy_parse_state,
          Parser::static_get_image_info, Parser::static_get_capabilities}
    , capabilities_{NVIMGCDCS_CAPABILITY_HOST_INPUT}
    , framework_(framework)
{
}

struct nvimgcdcsParserDesc* NvJpeg2kParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::canParse(bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    constexpr uint8_t JP2_SIGNATURE[] = {0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20, 0x0d, 0x0a, 0x87, 0x0a};
    constexpr uint8_t J2K_SIGNATURE[] = {0xff, 0x4f};
    size_t signature_size = sizeof(JP2_SIGNATURE);
    std::vector<unsigned char> signature_buffer(signature_size);
    size_t output_size = 0;
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    io_stream->read(io_stream->instance, &output_size, &signature_buffer[0], signature_size);
    if (output_size != signature_size) {
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    *result = !memcmp(signature_buffer.data(), JP2_SIGNATURE, sizeof(JP2_SIGNATURE)) ||
              !memcmp(signature_buffer.data(), J2K_SIGNATURE, sizeof(J2K_SIGNATURE));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::static_can_parse(void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg2k_parser_can_parse");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(result);
        XM_CHECK_NULL(code_stream);
        auto handle = reinterpret_cast<NvJpeg2kParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not check if nvjpeg2k can parse code stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpeg2kParserPlugin::Parser::Parser(const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework)
    : capabilities_(capabilities)
    , framework_(framework)
{
    XM_CHECK_NVJPEG2K(nvjpeg2kCreateSimple(&handle_));
}

NvJpeg2kParserPlugin::Parser::~Parser()
{
    try {
        XM_CHECK_NVJPEG2K(nvjpeg2kDestroy(handle_));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not destroy nvjpeg2k parser");
    }
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::create(nvimgcdcsParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcdcsParser_t>(new NvJpeg2kParserPlugin::Parser(capabilities_, framework_));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg2k_parser_create");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(parser);
        auto handle = reinterpret_cast<NvJpeg2kParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not create nvjpeg2k parser - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg2k_parser_destroy");
        XM_CHECK_NULL(parser);
        auto handle = reinterpret_cast<NvJpeg2kParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not destroy nvjpeg2k parser - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::Parser::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t NvJpeg2kParserPlugin::Parser::static_get_capabilities(
    nvimgcdcsParser_t parser, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg2k_get_capabilities");
        XM_CHECK_NULL(parser);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        auto handle = reinterpret_cast<NvJpeg2kParserPlugin::Parser*>(parser);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not retrive nvjpeg2k parser capabilites " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::Parser::createParseState(nvimgcdcsParseState_t* parse_state)
{
    auto par_state = new NvJpeg2kParserPlugin::ParseState();
    *parse_state = reinterpret_cast<nvimgcdcsParseState_t>(par_state);
    XM_CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&par_state->nvjpeg2k_stream_));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::Parser::static_create_parse_state(nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("nvjpeg2k_create_parse_state");
        XM_CHECK_NULL(parser);
        XM_CHECK_NULL(parse_state);
        auto handle = reinterpret_cast<NvJpeg2kParserPlugin::Parser*>(parser);
        return handle->createParseState(parse_state);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not create nvjpeg2k encode state " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::Parser::static_destroy_parse_state(nvimgcdcsParseState_t parse_state)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg2k_destroy_parse_state");
        XM_CHECK_NULL(parse_state);
        auto handle = reinterpret_cast<NvJpeg2kParserPlugin::ParseState*>(parse_state);
        if (handle->nvjpeg2k_stream_) {
            XM_CHECK_NVJPEG2K(nvjpeg2kStreamDestroy(handle->nvjpeg2k_stream_));
        }
        delete handle;
        return NVIMGCDCS_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not destroy jpeg2k parse state - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

// clang-format off
#define SAMPLE_DATA_TYPE(precision, sgn)                                                             \
    precision == 8  ? (sgn ? NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8 : NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8):   \
    precision == 16 ? (sgn ? NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16 : NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16):NVIMGCDCS_SAMPLE_DATA_TYPE_UNKNOWN;
// clang-format on

nvimgcdcsColorSpec_t nvjpeg2k_to_nvimgcdcs_color_space(nvjpeg2kColorSpace_t nvjpeg2k_color_space)
{
    switch (nvjpeg2k_color_space) {
    case NVJPEG2K_COLORSPACE_NOT_SUPPORTED:
        return NVIMGCDCS_COLORSPEC_UNSUPPORTED;
    case NVJPEG2K_COLORSPACE_UNKNOWN:
        return NVIMGCDCS_COLORSPEC_UNKNOWN;
    case NVJPEG2K_COLORSPACE_SRGB:
        return NVIMGCDCS_COLORSPEC_SRGB;
    case NVJPEG2K_COLORSPACE_GRAY:
        return NVIMGCDCS_COLORSPEC_GRAY;
    case NVJPEG2K_COLORSPACE_SYCC:
        return NVIMGCDCS_COLORSPEC_SYCC;
    default:
        return NVIMGCDCS_COLORSPEC_UNKNOWN;
    }
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::Parser::getImageInfo(nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    auto parse_state = reinterpret_cast<NvJpeg2kParserPlugin::ParseState*>(code_stream->parse_state);
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    size_t encoded_stream_data_size = 0;
    io_stream->size(io_stream->instance, &encoded_stream_data_size);
    const void* encoded_stream_data = nullptr;
    io_stream->raw_data(io_stream->instance, &encoded_stream_data);
    if (!encoded_stream_data) {
        if (parse_state->buffer_.size() != encoded_stream_data_size) {
            parse_state->buffer_.resize(encoded_stream_data_size);
            io_stream->seek(io_stream->instance, 0, SEEK_SET);
            size_t read_nbytes = 0;
            io_stream->read(io_stream->instance, &read_nbytes, &parse_state->buffer_[0], encoded_stream_data_size);
            if (read_nbytes != encoded_stream_data_size) {
                NVIMGCDCS_D_LOG_ERROR("Unexpected end-of-stream");
                return NVIMGCDCS_STATUS_BAD_CODESTREAM;
            }
        }
        encoded_stream_data = &parse_state->buffer_[0];
    }

    XM_CHECK_NVJPEG2K(nvjpeg2kStreamParse(handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size, false,
        false, parse_state->nvjpeg2k_stream_));

    nvjpeg2kImageInfo_t nvjpeg2k_image_info;
    XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(parse_state->nvjpeg2k_stream_, &nvjpeg2k_image_info));

    image_info->plane_info[0].width = nvjpeg2k_image_info.image_width;
    image_info->plane_info[0].height = nvjpeg2k_image_info.image_height;
    image_info->num_planes = nvjpeg2k_image_info.num_components;

    for (size_t p = 0; p < image_info->num_planes; p++) {

        nvjpeg2kImageComponentInfo_t component_info;
        XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(parse_state->nvjpeg2k_stream_, &component_info, p));

        image_info->plane_info[p].height = component_info.component_height;
        image_info->plane_info[p].width = component_info.component_width;
        image_info->plane_info[p].num_channels = 1;
        image_info->plane_info[p].sample_type = SAMPLE_DATA_TYPE(component_info.precision, component_info.sgn);
    }
    image_info->sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
    nvjpeg2kColorSpace_t color_space;
    XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetColorSpace(parse_state->nvjpeg2k_stream_, &color_space));
    image_info->color_spec = nvjpeg2k_to_nvimgcdcs_color_space(color_space);

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kParserPlugin::Parser::static_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg2k_parser_get_image_info");
        XM_CHECK_NULL(parser);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(image_info);
        auto handle = reinterpret_cast<NvJpeg2kParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not retrieve image info from jpeg2k code stream. - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

} // namespace nvjpeg2k
