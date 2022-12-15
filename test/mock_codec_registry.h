#include <gmock/gmock.h>
#include "../src/icodec_registry.h"
#include "../src/iimage_parser.h"
#include "../src/icodec.h"
#include <memory>

namespace nvimgcdcs { namespace test {

class MockCodecRegistry : public ICodecRegistry
{
  public:
    MOCK_METHOD(void, registerCodec, (std::unique_ptr<ICodec> codec), (override));
    MOCK_METHOD((const std::pair<ICodec*, std::unique_ptr<IImageParser>>), getCodecAndParser,(
        nvimgcdcsCodeStreamDesc_t code_stream), (const, override));
    MOCK_METHOD(ICodec*, getCodecByName, (const char* name) ,  (override));
};

}} // namespace nvimgcdcs::test