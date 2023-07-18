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

#ifdef NDEBUG
    #define NVIMGCDCS_SEVERITY NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO
#else
    #define NVIMGCDCS_SEVERITY NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE
#endif

#define NVIMGCDCS_LOG(framework, id, svr, type, msg)                                                                   \
    do {                                                                     \
        if (svr >= NVIMGCDCS_SEVERITY) {                                     \
            std::stringstream ss{};                                          \
            ss << msg;                                                       \
            std::string msg_str{ss.str()};                                                                             \
            const nvimgcdcsDebugMessageData_t data{                                                                    \
                NVIMGCDCS_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA, nullptr, msg_str.c_str(), 0, nullptr, id, NVIMGCDCS_VER}; \
            framework->log(framework->instance, svr, type, &data);                                                     \
        }                                                                    \
    } while (0)

#define NVIMGCDCS_LOG_TRACE(framework, id, ...) \
    NVIMGCDCS_LOG(framework, id, NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_LOG_DEBUG(framework, id, ...) \
    NVIMGCDCS_LOG(framework, id, NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_LOG_INFO(framework, id, ...) \
    NVIMGCDCS_LOG(framework, id, NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_LOG_WARNING(framework, id, ...) \
    NVIMGCDCS_LOG(framework, id, NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_LOG_ERROR(framework, id, ...) \
    NVIMGCDCS_LOG(framework, id, NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCDCS_LOG_FATAL(framework, id, ...) \
    NVIMGCDCS_LOG(framework, id, NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
