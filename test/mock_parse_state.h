#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nvimgcdcs_module.h>
#include "../src/iparse_state.h"

namespace nvimgcdcs { namespace test {

class MockParseState : public IParseState
{
  public:
    MOCK_METHOD(nvimgcdcsParseState_t, getInternalParseState,(), (override));
};

}} // namespace nvimgcdcs::test