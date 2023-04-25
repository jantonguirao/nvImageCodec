/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <nvimgcodecs.h>
#include "log.h"
#include "parsers/bmp.h"
#include "parsers/jpeg.h"
#include "parsers/jpeg2k.h"
#include "parsers/png.h"
#include "parsers/pnm.h"
#include "parsers/tiff.h"
#include "parsers/webp.h"

namespace nvimgcdcs {

nvimgcdcsStatus_t parsers_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    NVIMGCDCS_LOG_TRACE("parsers_extension_create");

    static auto bmp_parser_plugin = BMPParserPlugin();
    framework->registerParser(framework->instance, bmp_parser_plugin.getParserDesc());
    static auto jpeg_parser_plugin = JPEGParserPlugin();
    framework->registerParser(framework->instance, jpeg_parser_plugin.getParserDesc());
    static auto jpeg2k_parser_plugin = JPEG2KParserPlugin();
    framework->registerParser(framework->instance, jpeg2k_parser_plugin.getParserDesc());
    static auto png_parser_plugin = PNGParserPlugin();
    framework->registerParser(framework->instance, png_parser_plugin.getParserDesc());
    static auto pnm_parser_plugin = PNMParserPlugin();
    framework->registerParser(framework->instance, pnm_parser_plugin.getParserDesc());
    static auto tiff_parser_plugin = TIFFParserPlugin();
    framework->registerParser(framework->instance, tiff_parser_plugin.getParserDesc());
    static auto webp_parser_plugin = WebpParserPlugin();
    framework->registerParser(framework->instance, webp_parser_plugin.getParserDesc());

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t parsers_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("parsers_extension_destroy");

    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t parsers_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "parsers",    // id
     0x00000100,  // version

    parsers_extension_create,
    parsers_extension_destroy
};
// clang-format on

nvimgcdcsStatus_t get_parsers_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = parsers_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}

} // namespace nvimgcdcs