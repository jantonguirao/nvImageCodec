/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <nvimgcodec_version.h>
#include <nvimgcodec.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "module.h"
#include "image.h"
#include "decoder.h"
#include "encoder.h"
#include "decode_params.h"
#include "jpeg_encode_params.h"
#include "jpeg2k_encode_params.h"
#include "encode_params.h"
#include "color_spec.h"
#include "chroma_subsampling.h"
#include "jpeg2k_bitstream_type.h"
#include "jpeg2k_prog_order.h"
#include "backend_kind.h"
#include "backend_params.h"
#include "backend.h"
#include "image_buffer_kind.h"

#include <iostream>

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(nvimgcodec_impl, m)
{
    using namespace nvimgcodec;

    static Module module;

    m.doc() = R"pbdoc(

        nvImageCodec Python API reference

        This is the Python API reference for the NVIDIA® nvImageCodec library.
    )pbdoc";


    nvimgcodecProperties_t properties{NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES, 0};
    nvimgcodecGetProperties(&properties);
    std::stringstream ver_ss{};
    ver_ss << NVIMGCODEC_STREAM_VER(properties.version);
    m.attr("__version__") = ver_ss.str();
    m.attr("__cuda_version__") = properties.cudart_version;

    BackendKind::exportToPython(m);
    BackendParams::exportToPython(m);
    Backend::exportToPython(m);
    ColorSpec::exportToPython(m);
    ChromaSubsampling::exportToPython(m);
    ImageBufferKind::exportToPython(m);
    Jpeg2kBitstreamType::exportToPython(m);
    Jpeg2kProgOrder::exportToPython(m);
    DecodeParams::exportToPython(m);
    JpegEncodeParams::exportToPython(m);
    Jpeg2kEncodeParams::exportToPython(m);
    EncodeParams::exportToPython(m);
    Image::exportToPython(m);
    Decoder::exportToPython(m, module.instance_);
    Encoder::exportToPython(m, module.instance_);
    Module::exportToPython(m, module.instance_);
}
