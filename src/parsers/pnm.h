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
#include <vector>
#include <array>

namespace nvimgcdcs {

class PNMParserPlugin
{
  public:
    explicit PNMParserPlugin(const nvimgcdcsFrameworkDesc_t* framework);
    nvimgcdcsParserDesc_t* getParserDesc();

  private:
    struct Parser
    {
        Parser(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework);

        nvimgcdcsStatus_t getImageInfo(
            nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream);

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsParser_t parser);
        static nvimgcdcsStatus_t static_get_image_info(nvimgcdcsParser_t parser,
            nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t* code_stream);
        
        const char *plugin_id_;
        const nvimgcdcsFrameworkDesc_t *framework_;
    };

    nvimgcdcsStatus_t canParse(int* result, nvimgcdcsCodeStreamDesc_t* code_stream);
    nvimgcdcsStatus_t create(nvimgcdcsParser_t* parser);

    static nvimgcdcsStatus_t static_can_parse(
        void* instance, int* result, nvimgcdcsCodeStreamDesc_t* code_stream);
    static nvimgcdcsStatus_t static_create(void* instance, nvimgcdcsParser_t* parser);

    static constexpr const char* plugin_id_ = "pnm_parser";
    const nvimgcdcsFrameworkDesc_t* framework_;
    nvimgcdcsParserDesc_t parser_desc_;
};

nvimgcdcsStatus_t get_pnm_parser_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc);

}  // namespace nvimgcdcs