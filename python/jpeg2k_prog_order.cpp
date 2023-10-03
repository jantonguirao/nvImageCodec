/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "jpeg2k_prog_order.h"

namespace nvimgcodec {

void Jpeg2kProgOrder::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecJpeg2kProgOrder_t>(m, "Jpeg2kProgOrder")
        .value("LRCP", NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP)
        .value("RLCP", NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP)
        .value("RPCL", NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL)
        .value("PCRL", NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL)
        .value("CPRL", NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL)
        .export_values();
    // clang-format on
} ;


} // namespace nvimgcodec
