#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nvimgcdcs_module.h>
#include "../src/iimage_parser.h"

namespace nvimgcdcs { namespace test {

class MockImageParser : public IImageParser
{
  public:
    MOCK_METHOD(std::string, getParserId,(), (const, override));
    MOCK_METHOD(std::string, getCodecName,(), (const, override));
    MOCK_METHOD(void, getImageInfo,(
        nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageInfo_t* image_info) ,(override));
    MOCK_METHOD(std::unique_ptr<IParseState>, createParseState,(),(override));
};

}} // namespace nvimgcdcs::test