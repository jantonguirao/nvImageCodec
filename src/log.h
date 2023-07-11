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
#include <iostream>
#include <sstream>
#include <string>
#include "logger.h"

namespace nvimgcdcs {

#ifdef NDEBUG
    #define NVIMGCDCS_SEVERITY NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO
#else
    #define NVIMGCDCS_SEVERITY NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE
#endif

#define NVIMGCDCS_LOG_WITH_LOGGER(logger, svr, type, msg)  \
    do {                                       \
        if (svr >= NVIMGCDCS_SEVERITY) {       \
            std::stringstream ss{};            \
            ss << msg;                         \
            logger->log(svr, type, ss.str()); \
        }                                      \
    } while (0)

#define NVIMGCDCS_LOG(svr, type, msg) NVIMGCDCS_LOG_WITH_LOGGER(Logger::get(), svr, type, msg)
#define NVIMGCDCS_LOG_(svr, type, msg) NVIMGCDCS_LOG_WITH_LOGGER(logger_, svr, type, msg)

#ifdef NDEBUG

    #define NVIMGCDCS_LOG_TRACE(...)
    #define NVIMGCDCS_LOG_DEBUG(...)
    #define NVIMGCDCS_LOG_TRACE_(...)
    #define NVIMGCDCS_LOG_DEBUG_(...)

#else

    #define NVIMGCDCS_LOG_TRACE(...) \
        NVIMGCDCS_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

    #define NVIMGCDCS_LOG_DEBUG(...) \
        NVIMGCDCS_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

    #define NVIMGCDCS_LOG_TRACE_(...) \
        NVIMGCDCS_LOG_(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

    #define NVIMGCDCS_LOG_DEBUG_(...) \
        NVIMGCDCS_LOG_(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#endif

#define NVIMGCDCS_LOG_INFO(...) \
    NVIMGCDCS_LOG(logger, NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#define NVIMGCDCS_LOG_WARNING(...) \
    NVIMGCDCS_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#define NVIMGCDCS_LOG_ERROR(...) \
    NVIMGCDCS_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#define NVIMGCDCS_LOG_FATAL(...) \
    NVIMGCDCS_LOG(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#define NVIMGCDCS_LOG_INFO_(...) \
    NVIMGCDCS_LOG_(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#define NVIMGCDCS_LOG_WARNING_(...) \
    NVIMGCDCS_LOG_(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#define NVIMGCDCS_LOG_ERROR_(...) \
    NVIMGCDCS_LOG_(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

#define NVIMGCDCS_LOG_FATAL_(...) \
    NVIMGCDCS_LOG_(NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCDCS_DEBUG_MESSAGE_TYPE_GENERAL, __VA_ARGS__)

} //namespace nvimgcdcs