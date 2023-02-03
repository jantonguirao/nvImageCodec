# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.22)

project(nvimgcodecs_python CXX)

set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})
add_subdirectory(pybind11)

include_directories(
  ../include
)   
# find_package(nvimgcodecs REQUIRED)

set(CMAKE_CXX_STANDARD 20)

pybind11_add_module(nvimgcodecs_python MODULE
    main.cpp
)

target_include_directories(nvimgcodecs_python
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
)

set_target_properties(nvimgcodecs_python PROPERTIES
    OUTPUT_NAME nvimgcodecs
    DEBUG_POSTFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(nvimgcodecs_python
    PRIVATE
        nvimgcodecs_static
        CUDA::cudart_static
        -static-libstdc++
)

#target_compile_options(nvimgcodecs_python
#    PRIVATE
#        -Wno-deprecated-declarations
#        -Wsuggest-override
#)

include(GNUInstallDirs)

install(TARGETS nvimgcodecs_python
    LIBRARY DESTINATION python
)
