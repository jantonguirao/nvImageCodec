#include <nvimgcdcs_module.h>
#include <iostream>

static nvimgcdcsParserStatus_t example_parser_can_parse(
    bool* result, nvimgcdcsInputStreamDesc_t input_stream)
{
    std::cout << "example_parser_can_parse" << std::endl;
    *result = false;
    return NVIMGCDCS_PARSER_STATUS_SUCCESS;
}

static nvimgcdcsParserStatus_t example_parser_get_image_info(
    nvimgcdcsImageInfo_t* result, nvimgcdcsInputStreamDesc_t input_stream)
{
    std::cout << "example_parser_get_image_info" << std::endl;
    return NVIMGCDCS_PARSER_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsParserDesc example_parser = {
    NULL,               // instance    
    "example_parser",   // id
     0x00000100,        // version
    "raw",              // codec_type 

    example_parser_can_parse,
    example_parser_get_image_info
};
// clang-format on   