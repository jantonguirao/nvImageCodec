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

#include <nvimgcodec.h>
#include <vector>
#include <array>

namespace nvimgcodec {

class TIFFParserPlugin
{
  public:
    explicit TIFFParserPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecParserDesc_t* getParserDesc();

  private:
    struct Parser
    {
        Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework);
        
        nvimgcodecStatus_t getImageInfo(
            nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream);

        static nvimgcodecStatus_t static_destroy(nvimgcodecParser_t parser);
        static nvimgcodecStatus_t static_get_image_info(nvimgcodecParser_t parser,
            nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream);

        const char *plugin_id_;
        const nvimgcodecFrameworkDesc_t *framework_;
    };

    nvimgcodecStatus_t canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream);
    nvimgcodecStatus_t create(nvimgcodecParser_t* parser);

    static nvimgcodecStatus_t static_can_parse(
        void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream);
    static nvimgcodecStatus_t static_create(void* instance, nvimgcodecParser_t* parser);

    static constexpr const char* plugin_id_ = "tiff_parser";
    const nvimgcodecFrameworkDesc_t* framework_;
    nvimgcodecParserDesc_t parser_desc_;
};

nvimgcodecStatus_t get_tiff_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc);

}  // namespace nvimgcodec