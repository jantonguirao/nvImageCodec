#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "../src/iimage_parser.h"
#include "../src/iimage_parser_factory.h"

namespace nvimgcdcs {
namespace test {

class MockImageParserFactory : public IImageParserFactory
{
  public:
    MOCK_METHOD(std::string, getParserId, (), (const, override));
    MOCK_METHOD(std::string, getCodecName, (), (const, override));
    MOCK_METHOD(bool, canParse, (nvimgcdcsCodeStreamDesc_t code_stream), (override));
    MOCK_METHOD(std::unique_ptr<IImageParser>, createParser, (), (const, override));
};

}}