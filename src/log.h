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

#include <nvimgcodec.h>
#include <iostream>
#include <sstream>
#include <string>
#include "logger.h"

namespace nvimgcodec {

#ifdef NDEBUG
    #define NVIMGCODEC_SEVERITY NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO
#else
    #define NVIMGCODEC_SEVERITY NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE
#endif

#define NVIMGCODEC_LOG(logger, svr, type, msg) \
    do {                                      \
        if (svr >= NVIMGCODEC_SEVERITY) {      \
            std::stringstream ss{};           \
            ss << msg;                        \
            logger->log(svr, type, ss.str()); \
        }                                     \
    } while (0)

#ifdef NDEBUG

    #define NVIMGCODEC_LOG_TRACE(...)
    #define NVIMGCODEC_LOG_DEBUG(...)

#else

    #define NVIMGCODEC_LOG_TRACE(logger, ...) \
        NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

    #define NVIMGCODEC_LOG_DEBUG(logger, ...) \
        NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

#endif

#define NVIMGCODEC_LOG_INFO(logger, ...) \
    NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

#define NVIMGCODEC_LOG_WARNING(logger, ...) \
    NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

#define NVIMGCODEC_LOG_ERROR(logger, ...) \
    NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

#define NVIMGCODEC_LOG_FATAL(logger, ...) \
    NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)


} //namespace nvimgcodec