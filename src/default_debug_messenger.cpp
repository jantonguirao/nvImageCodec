
#include "default_debug_messenger.h"
#include <iostream>
#include <cassert>

#define TERM_NORMAL "\033[0m"
#define TERM_RED "\033[0;31m"
#define TERM_YELLOW "\033[0;33m"
#define TERM_GREEN "\033[0;32m"
#define TERM_MAGENTA "\033[1;35m"

namespace nvimgcodec {
DefaultDebugMessenger::DefaultDebugMessenger(uint32_t message_severity, uint32_t message_category)
    : desc_{NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSENGER_DESC, nullptr, message_severity, message_category,
          DefaultDebugMessenger::static_debug_callback, this}
{
}

int DefaultDebugMessenger::debugCallback(const nvimgcodecDebugMessageSeverity_t message_severity,
    const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* callback_data)
{
    switch (message_severity) {
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL:
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR:
        std::cerr << TERM_RED;
        break;
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING:
        std::cerr << TERM_YELLOW;
        break;
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO:
        std::cerr << TERM_GREEN;
        break;
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE:
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG:
        std::cerr << TERM_MAGENTA;
        break;
    default:
        break;
    }

    switch (message_severity) {
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR:
        std::cerr << "[ERROR] ";
        break;
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL:
        std::cerr << "[FATAL ERROR] ";
        break;
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING:
        std::cerr << "[WARNING] ";
        break;
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO:
        std::cerr << "[INFO] ";
        break;
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG:
        std::cerr << "[DEBUG] ";
        break;
    case NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE:
        std::cerr << "[TRACE] ";
        break;

    default:
        std::cerr << "UNKNOWN: ";
        break;
    }

    std::cerr << TERM_NORMAL;
    std::cerr << "[" << callback_data->codec_id << "] ";
    std::cerr << callback_data->message << std::endl;

    return 0;
}

int DefaultDebugMessenger::static_debug_callback(
    const nvimgcodecDebugMessageSeverity_t message_severity,
    const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* callback_data,
    void* user_data)
{
    assert(user_data);
    DefaultDebugMessenger* handle = reinterpret_cast<DefaultDebugMessenger*>(user_data);
    return handle->debugCallback(message_severity, message_category, callback_data);
}

} //namespace nvimgcodec