#include <gmock/gmock.h>
#include <memory>

#include "../src/iiostream_factory.h"
#include "../src/io_stream.h"

namespace nvimgcdcs { namespace test {

class MockIoStreamFactory : public IIoStreamFactory
{
  public:
    MOCK_METHOD(std::unique_ptr<IoStream>, createFileIoStream,
        (const std::string& file_name, bool read_ahead, bool use_mmap, bool to_write),
        (const, override));
    MOCK_METHOD(std::unique_ptr<IoStream>, createMemIoStream,(
            unsigned char* data, size_t size), (const, override));
};

}} // namespace nvimgcdcs::test