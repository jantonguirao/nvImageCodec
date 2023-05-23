/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodecs.h>
#include <sstream>
#include <string>

class Logger
{
  public:
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger) = delete;
    static Logger& get()
    {
        static Logger instance;
        return instance;
    }
    void log(const nvimgcdcsDebugMessageSeverity_t message_severity, const nvimgcdcsDebugMessageType_t message_type,
        const std::string& codec_id, uint32_t codec_version, const std::string& message)
    {
        if (log_func_) {
            const nvimgcdcsDebugMessageData_t data{
                NVIMGCDCS_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA, nullptr, message.c_str(), 0, nullptr, codec_id.c_str(), codec_version};
            log_func_(plugin_framework_instance_, message_severity, message_type, &data);
        }
    }

    void registerLogFunc(void* plugin_framework_instance, nvimgcdcsLogFunc_t log_func)
    {
        plugin_framework_instance_ = plugin_framework_instance;
        log_func_ = log_func;
    };

    void unregisterLogFunc()
    {
        plugin_framework_instance_ = nullptr;
        log_func_ = nullptr;
    };

  protected:
    Logger()
        : plugin_framework_instance_(nullptr)
        , log_func_(nullptr){};
    virtual ~Logger() {}

  private:
    void* plugin_framework_instance_;
    nvimgcdcsLogFunc_t log_func_;
};

#ifdef NDEBUG
    #define NVIMGCDCS_SEVERITY NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO
#else
    #define NVIMGCDCS_SEVERITY NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE
#endif

#define NVIMGCDCS_LOG(svr, type, codec_id, msg)                              \
    do {                                                                     \
        if (svr >= NVIMGCDCS_SEVERITY) {                                     \
            std::stringstream ss{};                                          \
            ss << msg;                                                       \
            Logger::get().log(svr, type, codec_id, NVIMGCDCS_VER, ss.str()); \
        }                                                                    \
    } while (0)

#define NVIMGCDCS_M_LOG(svr, type, msg) NVIMGCDCS_LOG(svr, type, "nvjpeg-module", msg)
#define NVIMGCDCS_P_LOG(svr, type, msg) NVIMGCDCS_LOG(svr, type, "nvjpeg-parser", msg)
#define NVIMGCDCS_E_LOG(svr, type, msg) NVIMGCDCS_LOG(svr, type, "nvjpeg-encoder", msg)
#define NVIMGCDCS_D_LOG(svr, type, msg) NVIMGCDCS_LOG(svr, type, "nvjpeg-decoder", msg)

#define NVIMGCDCS_LOG_TRACE(...) NVIMGCDCS_M_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_LOG_DEBUG(...) NVIMGCDCS_M_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_M_LOG_INFO(...) NVIMGCDCS_M_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_LOG_WARNING(...) \
    NVIMGCDCS_M_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_LOG_ERROR(...) NVIMGCDCS_M_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_LOG_FATAL(...) NVIMGCDCS_M_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#define NVIMGCDCS_P_LOG_TRACE(...) \
    NVIMGCDCS_P_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_P_LOG_DEBUG(...) \
    NVIMGCDCS_P_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_P_LOG_INFO(...) NVIMGCDCS_P_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_P_LOG_WARNING(...) \
    NVIMGCDCS_P_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_P_LOG_ERROR(...) \
    NVIMGCDCS_P_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_P_LOG_FATAL(...) \
    NVIMGCDCS_P_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#define NVIMGCDCS_E_LOG_TRACE(...) \
    NVIMGCDCS_E_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_E_LOG_DEBUG(...) \
    NVIMGCDCS_E_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_E_LOG_INFO(...) NVIMGCDCS_E_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_E_LOG_WARNING(...) \
    NVIMGCDCS_E_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_E_LOG_ERROR(...) \
    NVIMGCDCS_E_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_E_LOG_FATAL(...) \
    NVIMGCDCS_E_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#define NVIMGCDCS_D_LOG_TRACE(...) \
    NVIMGCDCS_D_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_D_LOG_DEBUG(...) \
    NVIMGCDCS_D_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_D_LOG_INFO(...) NVIMGCDCS_D_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_D_LOG_WARNING(...) \
    NVIMGCDCS_D_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_D_LOG_ERROR(...) \
    NVIMGCDCS_D_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_D_LOG_FATAL(...) \
    NVIMGCDCS_D_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)
