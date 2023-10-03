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
#include <sstream>
#include <string>

#ifdef NDEBUG
    #define NVIMGCODEC_SEVERITY NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO
#else
    #define NVIMGCODEC_SEVERITY NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE
#endif

#define NVIMGCODEC_LOG(framework, id, svr, type, msg)                                                                   \
    do {                                                                     \
        if (svr >= NVIMGCODEC_SEVERITY) {                                     \
            std::stringstream ss{};                                          \
            ss << msg;                                                       \
            std::string msg_str{ss.str()};                                                                             \
            const nvimgcodecDebugMessageData_t data{                                                                    \
                NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA, nullptr, msg_str.c_str(), 0, nullptr, id, NVIMGCODEC_VER}; \
            framework->log(framework->instance, svr, type, &data);                                                     \
        }                                                                    \
    } while (0)

#define NVIMGCODEC_LOG_TRACE(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCODEC_LOG_DEBUG(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCODEC_LOG_INFO(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCODEC_LOG_WARNING(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCODEC_LOG_ERROR(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCODEC_LOG_FATAL(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
