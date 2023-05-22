/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "nvjpeg_utils.h"
#include <sstream>
#include <string>

namespace nvjpeg {

unsigned int get_nvjpeg_flags(const char* module_name, const char* options) {

    // if available, we prefer this to be the default (it matches libjpeg implementation)
    bool fancy_upsampling = true;

    std::istringstream iss(options ? options : "");
    std::string token;
    while (std::getline(iss, token, ' ')) {
        std::string::size_type colon = token.find(':');
        std::string::size_type equal = token.find('=');
        if (colon == std::string::npos || equal == std::string::npos || colon > equal)
            continue;
        std::string module = token.substr(0, colon);
        if (module != "" && module != module_name)
            continue;
        std::string option = token.substr(colon + 1, equal - colon - 1);
        std::string value_str = token.substr(equal + 1);

        std::istringstream value(value_str);
        if (option == "fancy_upsampling") {
            value >> fancy_upsampling;
        }
    }

    int major = -1, minor = -1, version = -1;
    if (NVJPEG_STATUS_SUCCESS == nvjpegGetProperty(MAJOR_VERSION, &major) &&
        NVJPEG_STATUS_SUCCESS == nvjpegGetProperty(MINOR_VERSION, &minor)) {
        version = major * 1000 + minor;
    }

    unsigned int nvjpeg_flags = 0;
#ifdef NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION
    if (fancy_upsampling && version >= 12001) {
      nvjpeg_flags |= NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION;
    }
#endif
    return nvjpeg_flags;
}

}