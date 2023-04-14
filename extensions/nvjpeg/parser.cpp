#include "parser.h"
#include <nvimgcodecs.h>
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>

#include <nvtx3/nvtx3.hpp>

#include "errors_handling.h"
#include "parser.h"
#include "type_convert.h"
#include "log.h"
#include <nvjpeg.h>

namespace nvjpeg {

NvJpegParserPlugin::NvJpegParserPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : parser_desc_{NVIMGCDCS_STRUCTURE_TYPE_PARSER_DESC, NULL,
          this,            // instance
          "nvjpeg_parser", // id
          0x00000100,      // version
          "jpeg",          // codec_type
          static_can_parse, static_create, Parser::static_destroy,
          Parser::static_create_parse_state, Parser::static_destroy_parse_state,
          Parser::static_get_image_info, Parser::static_get_capabilities}
    , capabilities_{
        NVIMGCDCS_CAPABILITY_ORIENTATION,
        NVIMGCDCS_CAPABILITY_ROI,
        NVIMGCDCS_CAPABILITY_HOST_INPUT,
        NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT,
        NVIMGCDCS_CAPABILITY_LAYOUT_PLANAR,
        NVIMGCDCS_CAPABILITY_LAYOUT_INTERLEAVED
        }
    , framework_(framework)
{
}

struct nvimgcdcsParserDesc* NvJpegParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcdcsStatus_t NvJpegParserPlugin::canParse(bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    constexpr uint8_t JPG_SIGNATURE[] = {0xff, 0xd8};
    constexpr size_t signature_size = sizeof(JPG_SIGNATURE);
    std::vector<unsigned char> signature_buffer(signature_size);
    size_t output_size = 0;
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    io_stream->read(io_stream->instance, &output_size, &signature_buffer[0], signature_size);
    if (output_size != signature_size) {
        *result = false;
        NVIMGCDCS_P_LOG_WARNING("Wrong size during reading from io_stream");
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    *result = !memcmp(signature_buffer.data(), JPG_SIGNATURE, sizeof(JPG_SIGNATURE));

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegParserPlugin::static_can_parse(
    void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg_parser_can_parse");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(result);
        XM_CHECK_NULL(code_stream);
        auto handle = reinterpret_cast<NvJpegParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not check if nvjpeg can parse code stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpegParserPlugin::Parser::Parser(const std::vector<nvimgcdcsCapability_t>& capabilities,
    const nvimgcdcsFrameworkDesc_t framework)
    : capabilities_(capabilities)
    , framework_(framework)
{
    XM_CHECK_NVJPEG(nvjpegCreateSimple(&handle_));
}

NvJpegParserPlugin::Parser::~Parser()
{
    try {
        XM_CHECK_NVJPEG(nvjpegDestroy(handle_));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not destroy nvjpe parser");
    }
}

nvimgcdcsStatus_t NvJpegParserPlugin::create(nvimgcdcsParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcdcsParser_t>(
        new NvJpegParserPlugin::Parser(capabilities_, framework_));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegParserPlugin::static_create(void* instance, nvimgcdcsParser_t* parser)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg_parser_create");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(parser);
        auto handle = reinterpret_cast<NvJpegParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not create nvjpeg parser - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegParserPlugin::Parser::static_destroy(nvimgcdcsParser_t parser)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg_parser_destroy");
        XM_CHECK_NULL(parser);
        auto handle = reinterpret_cast<NvJpegParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not destroy nvjpeg parser - " << e.what());
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegParserPlugin::Parser::getCapabilities(
    const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t NvJpegParserPlugin::Parser::static_get_capabilities(
    nvimgcdcsParser_t parser, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg_get_capabilities");
        XM_CHECK_NULL(parser);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        auto handle = reinterpret_cast<NvJpegParserPlugin::Parser*>(parser);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not retrive nvjpge parser capabilites - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t NvJpegParserPlugin::Parser::createParseState(nvimgcdcsParseState_t* parse_state)
{
    auto par_state = new NvJpegParserPlugin::ParseState();
    *parse_state = reinterpret_cast<nvimgcdcsParseState_t>(par_state);
    XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &par_state->nvjpeg_stream_));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegParserPlugin::Parser::static_create_parse_state(
    nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("nvjpeg_create_parse_state");
        XM_CHECK_NULL(parser);
        XM_CHECK_NULL(parse_state);
        auto handle = reinterpret_cast<NvJpegParserPlugin::Parser*>(parser);
        return handle->createParseState(parse_state);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not create nvjpeg encode state - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t NvJpegParserPlugin::Parser::static_destroy_parse_state(
    nvimgcdcsParseState_t parse_state)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg_destroy_parse_state");
        XM_CHECK_NULL(parse_state);
        auto handle = reinterpret_cast<NvJpegParserPlugin::ParseState*>(parse_state);
        if (handle->nvjpeg_stream_) {
            XM_CHECK_NVJPEG(nvjpegJpegStreamDestroy(handle->nvjpeg_stream_));
        }
        delete handle;
        return NVIMGCDCS_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not destroy nvjpeg parse state - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}


nvimgcdcsStatus_t NvJpegParserPlugin::Parser::getImageInfo(
    nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    NVIMGCDCS_P_LOG_TRACE("jpeg_parser_get_image_info");
    nvtx3::scoped_range marker{"getImageInfo"};
    try {
        size_t encoded_stream_data_size = 0;
        auto parse_state =
            reinterpret_cast<NvJpegParserPlugin::ParseState*>(code_stream->parse_state);
        nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
        io_stream->size(io_stream->instance, &encoded_stream_data_size);
        const void* encoded_stream_data = nullptr;
        io_stream->raw_data(io_stream->instance, &encoded_stream_data);
        if (!encoded_stream_data) {
            parse_state->buffer_.resize(encoded_stream_data_size);
            io_stream->seek(io_stream->instance, 0, SEEK_SET);
            size_t read_nbytes = 0;
            io_stream->read(io_stream->instance, &read_nbytes, &parse_state->buffer_[0], encoded_stream_data_size);
            if (read_nbytes != encoded_stream_data_size) {
                NVIMGCDCS_P_LOG_ERROR("Unexpected end-of-stream");
                return NVIMGCDCS_STATUS_BAD_CODESTREAM;
            }
            encoded_stream_data = &parse_state->buffer_[0];
        }
        assert(encoded_stream_data != nullptr);

        XM_CHECK_NVJPEG(nvjpegJpegStreamParseHeader(
            handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size, parse_state->nvjpeg_stream_));

         XM_CHECK_NVJPEG(nvjpegJpegStreamGetFrameDimensions(
            parse_state->nvjpeg_stream_, &image_info->plane_info[0].width, &image_info->plane_info[0].height));

        XM_CHECK_NVJPEG(nvjpegJpegStreamGetComponentsNum(parse_state->nvjpeg_stream_, &image_info->num_planes));

        image_info->sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        image_info->color_spec = NVIMGCDCS_COLORSPEC_UNKNOWN;
        auto sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8; 
        for (uint32_t p = 0; p < image_info->num_planes; ++p) {
            XM_CHECK_NVJPEG(nvjpegJpegStreamGetComponentDimensions(
                parse_state->nvjpeg_stream_, p, &image_info->plane_info[p].width, &image_info->plane_info[p].height));
            image_info->plane_info[p].num_channels = 1;
            image_info->plane_info[p].sample_type = sample_type;
        }
        nvjpegChromaSubsampling_t nvjpeg_css;
        XM_CHECK_NVJPEG(nvjpegJpegStreamGetChromaSubsampling(parse_state->nvjpeg_stream_, &nvjpeg_css));
        image_info->chroma_subsampling = nvjpeg_to_nvimgcdcs_css(nvjpeg_css);

        nvjpegExifOrientation_t orientation_flag;
        nvjpegJpegStreamGetExifOrientation(parse_state->nvjpeg_stream_, &orientation_flag);
        image_info->orientation = exif_to_nvimgcdcs_orientation(orientation_flag);

        nvimgcdcsJpegImageInfo_t* jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(image_info->next);
        while (jpeg_image_info && jpeg_image_info->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
            jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(jpeg_image_info->next);
        if (jpeg_image_info) {
            nvjpegJpegEncoding_t jpeg_encoding;
            XM_CHECK_NVJPEG(nvjpegJpegStreamGetJpegEncoding(parse_state->nvjpeg_stream_, &jpeg_encoding));
            jpeg_image_info->encoding = nvjpeg_to_nvimgcdcs_encoding(jpeg_encoding);
        }

        }
        catch (const std::runtime_error& e)
        {
            NVIMGCDCS_P_LOG_ERROR("Could not retrieve image info from jpeg stream - " << e.what());
            return NVIMGCDCS_STATUS_INTERNAL_ERROR;
        }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegParserPlugin::Parser::static_get_image_info(nvimgcdcsParser_t parser,
    nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream)
{
    try {
        NVIMGCDCS_P_LOG_TRACE("jpeg_parser_get_image_info");
        XM_CHECK_NULL(parser);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(image_info);
        auto handle = reinterpret_cast<NvJpegParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_P_LOG_ERROR("Could not retrieve image info from jpeg code stream - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

} // namespace nvjpeg
