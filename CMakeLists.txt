cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(NVIMGCODECS_PROJECT_NAME "nvimgcodecs")
set(NVIMGCODECS_VERSION "0.1.0")

if (NVIMGCODECS_FLAVOR)
    set(NVIMGCODECS_FLAVOR "${NVIMGCODECS_FLAVOR} ")  # could be nightly, weekly, etc if necessary
    set(NVIMGCODECS_FLAVOR_MINUS "-${NVIMGCODECS_FLAVOR}")  # -nightly, -weekly, etc
endif()

if (NOT ARCH)
    set(ARCH "x86_64")
endif()

if (NOT NVIMGCODECS_WHL_PLATFORM_NAME)
    set(NVIMGCODECS_WHL_PLATFORM_NAME "manylinux2014_${ARCH}")
endif()

if(NOT BUILD_ID)
    set (BUILD_ID 0)
endif()

set(NVIMGCODECS_VERSION_WITH_BUILD "${NVIMGCODECS_VERSION}.${BUILD_ID}")
project(${NVIMGCODECS_PROJECT_NAME} VERSION ${NVIMGCODECS_VERSION_WITH_BUILD} LANGUAGES CXX CUDA)
set(CUDA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

# Include custom cmake modules/scripts
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

include(Utils)
parse_cuda_version(${CUDA_VERSION} CUDA_VERSION_MAJOR CUDA_VERSION_MINOR CUDA_VERSION_PATCH CUDA_VERSION_SHORT CUDA_VERSION_SHORT_DIGIT_ONLY)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  # XXX: cmake appends /MT after /MD that produces D9025 warnings about overriding flags, hence...
  string(REGEX REPLACE "/MD " "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  string(REGEX REPLACE "/MDd " "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

option(BUILD_LIBRARY "Build library" ON)
option(BUILD_TEST "Build tests" ON)
option(BUILD_SAMPLES "Build samples" ON)
option(BUILD_CVCUDA_SAMPLES "Build CVCUDA samples" OFF)
option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_EXTENSIONS "Build extensions modules" ON)
option(BUILD_NVJPEG_EXT "Build nvjpeg extensions module" ON)
option(BUILD_NVJPEG2K_EXT "Build nvjpeg2k extensions module" ON)
option(BUILD_NVBMP_EXT "Build nvbmp extensions module" ON)
option(BUILD_NVPNM_EXT "Build nvpnm extensions module" ON)
option(BUILD_LIBJPEG_TURBO_EXT "Build libjpeg-turbo extensions module" ON)
option(BUILD_LIBTIFF_EXT "Build libtiff extensions module" ON)
option(BUILD_OPENCV_EXT "Build opencv extensions module" ON)
option(BUILD_PYTHON "Build Python binding" ON)
option(BUILD_WHEEL "Build python wheel package" ON)
option(WITH_DYNAMIC_NVJPEG "Dynamically loads nvjpeg at runtime" ON)
option(WITH_DYNAMIC_NVJPEG2K "Dynamically loads nvjpeg2k at runtime" ON)
option(WITH_DYNAMIC_NPP "Dynamically loads npp at runtime" ON)

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYS_NAME)

if(WIN32)
set(CPACK_GENERATOR "ZIP")
else()
set(CPACK_GENERATOR "TGZ")
endif()

set(CPACK_PACKAGE_NAME "nvimgcodecs")
set(CPACK_PACKAGE_VENDOR "NVIDIA CORPORATION")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU Accelerated Image Codecs library")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${PROJECT_VERSION})
set(CPACK_STRIP_FILES TRUE)

include (CPack)


# Targeted CUDA Architectures
# ALL: 30;35;50;60;70;75;82
set(CUDA_TARGET_ARCHS 52;60;61;70;75;80 CACHE
    STRING "List of targeted CUDA architectures (compute capabilities), for example \"30;35;50\""
)
message(STATUS "Targeted CUDA architectures: ${CUDA_TARGET_ARCHS}")

set(CUDA_PTX_ARCHS 80 CACHE
    STRING "List of targeted CUDA architectures (compute capabilities), for example \"30;35;50\""
)
message(STATUS "Targeted CUDA PTX architectures: ${CUDA_PTX_ARCHS}")

# Helper functions for setting CUDA architectures
include(SetCUDAArchitectures)

if(WIN32)
include(third_party/generate_product_version)

generate_product_version(ProductVersionFiles
    NAME "NVIDIA nvImageCodecs library"
    VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR}
    VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR}
    VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH}
    VERSION_REVISION ${CMAKE_PROJECT_VERSION_TWEAK}
    COMPANY_NAME NVIDIA Corporation)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/test
    )

# Find all dependencies
include(Dependencies)

add_subdirectory(external)

if(BUILD_LIBRARY)
add_subdirectory(src)
endif()

if(BUILD_SAMPLES)
add_subdirectory(example)
endif()

if(BUILD_PYTHON)
add_subdirectory(python)
endif()

#if(BUILD_DOCS)
#add_subdirectory(docs)
#endif()

if(BUILD_TEST)
enable_testing()
add_subdirectory(test)
endif()

set (NVIMGCDCS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)
if(BUILD_EXTENSIONS)
    add_subdirectory(extensions)
endif()