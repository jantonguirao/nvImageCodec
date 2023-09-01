/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <cstdlib>
#include <iostream>
#include <string>

#include <nvimgcodecs.h>
#include "image.h"
#include "module.h"

namespace nvimgcdcs {

uint32_t verbosity2severity(int verbose)
{
    uint32_t result = 0;
    if (verbose >= 1)
        result |= NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL | NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR;
    if (verbose >= 2)
        result |= NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING;
    if (verbose >= 3)
        result |= NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_INFO;
    if (verbose >= 4)
        result |= NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEBUG;
    if (verbose >= 5)
        result |= NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_TRACE;

    return result;
}

Module::Module()
{
    int verbosity = 1;
    char* v = std::getenv("PYNVIMGCODECS_VERBOSITY");
    try {
        if (v) {
            verbosity = std::stoi(v);
        }
    } catch (std::invalid_argument const& ex) {
        std::cerr << "[Warning] PYNVIMGCODECS_VERBOSITY has wrong value " << std::endl;
    } catch (std::out_of_range const& ex) {
        std::cerr << "[Warning] PYNVIMGCODECS_VERBOSITY has out of range value " << std::endl;
    }

    nvimgcdcsInstanceCreateInfo_t instance_create_info{NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0};
    instance_create_info.load_builtin_modules = 1;
    instance_create_info.load_extension_modules = 1;
    instance_create_info.default_debug_messenger = verbosity > 0 ? 1 : 0;
    instance_create_info.message_severity = verbosity2severity(verbosity);
    instance_create_info.message_category = NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_ALL;
    nvimgcdcsInstanceCreate(&instance_, &instance_create_info);
}

Module ::~Module()
{
    nvimgcdcsInstanceDestroy(instance_);
}

void Module::exportToPython(py::module& m, nvimgcdcsInstance_t instance)
{
    m.def(
         "as_image", [instance](py::handle source, intptr_t cuda_stream) -> Image { return Image(instance, source.ptr(), cuda_stream); },
         R"pbdoc(
            Wraps an external buffer as an image and ties the buffer lifetime to the image

            Args:
                source: Input DLPpack tensor which is encapsulated in a PyCapsule object or other object with __cuda_array_interface__ 
                or __dlpack__ and __dlpack_device__ methods.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place in created Image.

            Returns:
                nvimgcodecs.Image

        )pbdoc",
         "source"_a, "cuda_stream"_a = 0, py::keep_alive<0, 1>())
        .def(
            "from_dlpack",
            [instance](py::handle source, intptr_t cuda_stream) -> Image { return Image(instance, source.ptr(), cuda_stream); },
            R"pbdoc(
            Zero-copy conversion from a DLPack tensor to a image. 

            Args:
                source: Input DLPpack tensor which is encapsulated in a PyCapsule object or other (array) object with __dlpack__ 
                and __dlpack_device__ methods.
                
                cuda_stream: An optional cudaStream_t represented as a Python integer, upon which synchronization must take place in created Image.
            
            Returns:
                nvimgcodecs.Image

            )pbdoc",
            "source"_a, "cuda_stream"_a = 0, py::keep_alive<0, 1>());
}

} // namespace nvimgcdcs
