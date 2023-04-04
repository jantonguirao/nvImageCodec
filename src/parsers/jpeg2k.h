/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodecs.h>

namespace nvimgcdcs {

class JPEG2KParserPlugin
{
  public:
    explicit JPEG2KParserPlugin();
    struct nvimgcdcsParserDesc* getParserDesc();

  private:
    struct Parser
    {
        Parser();

        nvimgcdcsStatus_t getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size);
        nvimgcdcsStatus_t createParseState(nvimgcdcsParseState_t* parse_state);
        nvimgcdcsStatus_t destroyParseSate(nvimgcdcsParseState_t parse_state);
        nvimgcdcsStatus_t getImageInfo(
            nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream);

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsParser_t parser);
        static nvimgcdcsStatus_t static_get_capabilities(
            nvimgcdcsParser_t parser, const nvimgcdcsCapability_t** capabilities, size_t* size);
        static nvimgcdcsStatus_t static_create_parse_state(
            nvimgcdcsParser_t parser, nvimgcdcsParseState_t* parse_state);
        static nvimgcdcsStatus_t static_destroy_parse_state(nvimgcdcsParseState_t parse_state);
        static nvimgcdcsStatus_t static_get_image_info(nvimgcdcsParser_t parser,
            nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream);
    };

    nvimgcdcsStatus_t canParse(bool* result, nvimgcdcsCodeStreamDesc_t code_stream);
    nvimgcdcsStatus_t create(nvimgcdcsParser_t* parser);

    static nvimgcdcsStatus_t static_can_parse(
        void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream);
    static nvimgcdcsStatus_t static_create(void* instance, nvimgcdcsParser_t* parser);

    struct nvimgcdcsParserDesc parser_desc_;
};

nvimgcdcsStatus_t get_jpeg2k_parser_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc);

}  // namespace nvimgcdcs