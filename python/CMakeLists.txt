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

include(Utils)
include(GNUInstallDirs)

if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fvisibility=hidden")
endif()

set(PY_NVIMGCDCS_SRCS
    main.cpp
    module.cpp
    image.cpp
    decoder.cpp
    encoder.cpp
    decode_params.cpp
    jpeg_encode_params.cpp
    jpeg2k_encode_params.cpp
    encode_params.cpp
    color_spec.cpp
    chroma_subsampling.cpp
    jpeg2k_prog_order.cpp
    jpeg2k_bitstream_type.cpp
    error_handling.cpp
    backend_kind.cpp
    backend_params.cpp
    backend.cpp
    dlpack_utils.cpp)

build_per_python_lib(nvimgcodecs_python
    OUTPUT_NAME nvimgcodecs_impl
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/nvidia/nvimgcodecs
    PUBLIC_LIBS nvimgcodecs
    PRIV_LIBS CUDA::cudart_static -static-libstdc++
    SRC ${PY_NVIMGCDCS_SRCS})

# TODO(janton)
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/nvidia" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")
make_directory("${CMAKE_CURRENT_BINARY_DIR}/nvidia/nvimgcodecs/extensions")
configure_file("__init__.py.in" "${CMAKE_CURRENT_BINARY_DIR}/nvidia/nvimgcodecs/__init__.py")
configure_file("setup.py.in" "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/MANIFEST.in" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${PROJECT_SOURCE_DIR}/LICENSE.txt" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_custom_target(copy_libs_to_python_dir ALL DEPENDS nvimgcodecs_python)
add_dependencies(copy_libs_to_python_dir
    nvimgcodecs
    nvbmp_ext
    nvpnm_ext)

if (BUILD_NVJPEG_EXT)
    add_dependencies(copy_libs_to_python_dir nvjpeg_ext)
endif()

if (BUILD_NVJPEG2K_EXT)
    add_dependencies(copy_libs_to_python_dir nvjpeg2k_ext)
endif()

if(BUILD_LIBJPEG_TURBO_EXT)
    add_dependencies(copy_libs_to_python_dir libjpeg_turbo_ext)
endif()

if(BUILD_LIBTIFF_EXT)
    add_dependencies(copy_libs_to_python_dir libtiff_ext)
endif()

if(BUILD_OPENCV_EXT)
    add_dependencies(copy_libs_to_python_dir opencv_ext)
endif()

if(UNIX)
    add_custom_command(
        TARGET copy_libs_to_python_dir
        COMMAND cp "${PROJECT_BINARY_DIR}/src/libnvimgcodecs.so.0" "${CMAKE_CURRENT_BINARY_DIR}/nvidia/nvimgcodecs" &&
        cp "${PROJECT_BINARY_DIR}/extensions/*/*.so*" "${CMAKE_CURRENT_BINARY_DIR}/nvidia/nvimgcodecs/extensions" &&
        cp -r "${PROJECT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/nvidia/nvimgcodecs"
    )
else()
    add_custom_command(
        TARGET copy_libs_to_python_dir
        COMMAND echo "TODO copy_libs_to_python_dir")
endif()

if(BUILD_WHEEL)
    set(WHEEL_FILENAME "nvidia_nvimgcodecs_cuda${CUDA_VERSION_MAJOR}0-${NVIMGCODECS_VERSION}-py3-none-${NVIMGCODECS_WHL_PLATFORM_NAME}.whl")
    set(WHEEL_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(WHEEL_FILEPATH "${WHEEL_DIR}/${WHEEL_FILENAME}")

    # Note: not adding this target to ALL
    add_custom_target(wheel DEPENDS copy_libs_to_python_dir)

    if(UNIX)
        add_custom_command(
            TARGET wheel
            COMMAND ${PYTHON_EXECUTABLE} -m pip wheel --build-option --verbose --build-option --compression=stored --build-option --python-tag=py3 --build-option --plat-name=${NVIMGCODECS_WHL_PLATFORM_NAME} ${CMAKE_CURRENT_BINARY_DIR} &&
            ${CMAKE_CURRENT_SOURCE_DIR}/bundle-wheel.sh ${WHEEL_FILEPATH} NO NO ${WHEEL_FILENAME} /usr/local ${WHEEL_DIR} YES)

    else()
        add_custom_command(
            TARGET wheel
            COMMAND ${PYTHON_EXECUTABLE} -m pip wheel --build-option --verbose --build-option --compression=stored --build-option --python-tag=py3 ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()
endif()
