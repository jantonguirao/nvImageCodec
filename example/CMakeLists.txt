if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Build as a stand-alone samples project
    cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
    project(nvimgcodecs_samples LANGUAGES CXX CUDA)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    option(BUILD_CVCUDA_SAMPLES "Build CVCUDA samples" OFF)

    find_package(CUDAToolkit REQUIRED)

    list(APPEND CMAKE_PREFIX_PATH "/opt/nvidia/nvimgcodecs")
    find_package(nvimgcodecs CONFIG REQUIRED)

    include_directories(${nvimgcodecs_INCLUDE_DIR})
    link_directories(${nvimgcodecs_LIB_DIR})

    if(NOT DEFINED NVIMGCDCS_INCLUDE_DIR)
        set(NVIMGCDCS_INCLUDE_DIR ${nvimgcodecs_INCLUDE_DIR})
    endif()
    include_directories(${NVIMGCDCS_INCLUDE_DIR})
else()
    # Build as a part of nvImageCodecs project
    install(DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        ${CMAKE_CURRENT_SOURCE_DIR}/nvimtrans
        ${CMAKE_CURRENT_SOURCE_DIR}/nvimproc
        ${CMAKE_CURRENT_SOURCE_DIR}/python
        DESTINATION samples
        COMPONENT samples
    )

    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        ${CMAKE_CURRENT_SOURCE_DIR}/NOTICE.md
        ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
        DESTINATION samples
        COMPONENT samples
    )
endif()

add_subdirectory(nvimtrans)

if(BUILD_CVCUDA_SAMPLES)
    add_subdirectory(nvimproc)
endif()

