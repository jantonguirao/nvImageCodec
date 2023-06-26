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

class TIFFParserPlugin
{
  public:
    explicit TIFFParserPlugin();
    struct nvimgcdcsParserDesc* getParserDesc();

  private:
    struct Parser
    {
        Parser();

        
        nvimgcdcsStatus_t getImageInfo(
            nvimgcdcsImageInfo_t* image_info, nvimgcdcsCodeStreamDesc_t code_stream);

        static nvimgcdcsStatus_t static_destroy(nvimgcdcsParser_t parser);

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

nvimgcdcsStatus_t get_tiff_parser_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc);

}  // namespace nvimgcdcs