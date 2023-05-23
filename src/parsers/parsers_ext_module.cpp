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
#include "exception.h"
#include "log.h"
#include "parsers/bmp.h"
#include "parsers/jpeg.h"
#include "parsers/jpeg2k.h"
#include "parsers/png.h"
#include "parsers/pnm.h"
#include "parsers/tiff.h"
#include "parsers/webp.h"

namespace nvimgcdcs {

class ParsersExtension
{
  public:
    explicit ParsersExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
    {
        framework->registerParser(framework->instance, bmp_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
        framework->registerParser(framework->instance, jpeg_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
        framework->registerParser(framework->instance, jpeg2k_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
        framework->registerParser(framework->instance, png_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
        framework->registerParser(framework->instance, pnm_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
        framework->registerParser(framework->instance, tiff_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
        framework->registerParser(framework->instance, webp_parser_plugin_.getParserDesc(), NVIMGCDCS_PRIORITY_NORMAL);
    }
    ~ParsersExtension() {
        framework_->unregisterParser(framework_->instance, bmp_parser_plugin_.getParserDesc());
        framework_->unregisterParser(framework_->instance, jpeg_parser_plugin_.getParserDesc());
        framework_->unregisterParser(framework_->instance, jpeg2k_parser_plugin_.getParserDesc());
        framework_->unregisterParser(framework_->instance, png_parser_plugin_.getParserDesc());
        framework_->unregisterParser(framework_->instance, pnm_parser_plugin_.getParserDesc());
        framework_->unregisterParser(framework_->instance, tiff_parser_plugin_.getParserDesc());
        framework_->unregisterParser(framework_->instance, webp_parser_plugin_.getParserDesc());
    }

  private:
    const nvimgcdcsFrameworkDesc_t framework_;
    BMPParserPlugin bmp_parser_plugin_;
    JPEGParserPlugin jpeg_parser_plugin_;
    JPEG2KParserPlugin jpeg2k_parser_plugin_;
    PNGParserPlugin png_parser_plugin_;
    PNMParserPlugin pnm_parser_plugin_;
    TIFFParserPlugin tiff_parser_plugin_;
    WebpParserPlugin webp_parser_plugin_;
};

nvimgcdcsStatus_t parsers_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    NVIMGCDCS_LOG_TRACE("parsers_extension_create");
    try {
        CHECK_NULL(framework)
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new ParsersExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t parsers_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("parsers_extension_destroy");
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcdcs::ParsersExtension*>(extension);
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t parsers_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "nvimgcdcs_builtin_parsers",
    NVIMGCDCS_VER,
    NVIMGCDCS_EXT_API_VER, 

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