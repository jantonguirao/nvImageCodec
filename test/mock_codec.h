#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "../src/icodec.h"
#include "../src/iimage_parser.h"
#include "../src/iimage_decoder.h"
#include "../src/iimage_encoder.h"
#include "../src/iimage_parser_factory.h"
#include "../src/iimage_decoder_factory.h"
#include "../src/iimage_encoder_factory.h"

#include <memory>

namespace nvimgcdcs { namespace test {


class MockCodec : public ICodec
{
  public:
    MOCK_METHOD(const std::string&, name, (), (const, override));
    MOCK_METHOD(std::unique_ptr<IImageParser>, createParser,
        (nvimgcdcsCodeStreamDesc_t code_stream), (const, override));
    MOCK_METHOD(std::unique_ptr<IImageDecoder>, createDecoder,
        (nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsDecodeParams_t* params),
        (const, override));
    MOCK_METHOD(std::unique_ptr<IImageEncoder>, createEncoder,
        (nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsEncodeParams_t* params),
        (const, override));
    MOCK_METHOD(void, registerParserFactory,
        (std::unique_ptr<IImageParserFactory> factory, float priority), (override));
    MOCK_METHOD(void, registerEncoderFactory,
        (std::unique_ptr<IImageEncoderFactory> factory, float priority), (override));
    MOCK_METHOD(void, registerDecoderFactory,
        (std::unique_ptr<IImageDecoderFactory> factory, float priority), (override));
};

}} // namespace nvimgcdcs::test