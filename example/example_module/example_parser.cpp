#include <nvimgcdcs_module.h>
#include <iostream>

static nvimgcdcsParserStatus_t example_parser_can_parse(
    void* instance, bool* result, nvimgcdcsCodeStreamDesc_t code_stream)
{
    std::cout << "example_parser_can_parse" << std::endl;
    *result = false;
    return NVIMGCDCS_PARSER_STATUS_SUCCESS;
}

nvimgcdcsParserStatus_t example_parser_create(void* instance, nvimgcdcsParser_t* parser)
{
    return NVIMGCDCS_PARSER_STATUS_SUCCESS;
}

nvimgcdcsParserStatus_t example_parser_destroy(nvimgcdcsParser_t parser)
{
    return NVIMGCDCS_PARSER_STATUS_SUCCESS;
}

static nvimgcdcsParserStatus_t example_parser_get_image_info(
    nvimgcdcsParser_t parser, nvimgcdcsImageInfo_t* result, nvimgcdcsCodeStreamDesc_t code_stream)
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
    example_parser_create,
    example_parser_destroy,
    example_parser_get_image_info
};
// clang-format on   