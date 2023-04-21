/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "builtin_modules.h"
#include "parsers/parsers_ext_module.h"
#include "exception.h"
#include <vector>

namespace nvimgcdcs {

const std::vector<nvimgcdcsExtensionDesc_t>& get_builtin_modules() {
    static std::vector<nvimgcdcsExtensionDesc_t> builtin_modules_vec;
    if (builtin_modules_vec.empty()) {
        builtin_modules_vec.push_back({NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC, nullptr});
        nvimgcdcsStatus_t ret = get_parsers_extension_desc(&builtin_modules_vec.back());
        if (ret != NVIMGCDCS_STATUS_SUCCESS)
            throw Exception(INTERNAL_ERROR, "Failed to load parsers extension");
    }
    return builtin_modules_vec;
}

} // namespace nvimgcdcs
