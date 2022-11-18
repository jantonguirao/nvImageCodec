cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
if(NOT BUILD_ID)
set (BUILD_ID 0)
endif()

project(nvimgcdcs VERSION 0.1.0.${BUILD_ID} LANGUAGES CXX CUDA)

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
option(BUILD_EXAMPLE "Build example" ON)
option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_EXTENSIONS "Build extensions modules" ON)
option(BUILD_PYTHON "Build Python binding" ON)



string(TOLOWER ${CMAKE_SYSTEM_NAME} SYS_NAME)

if(WIN32)
set(CPACK_GENERATOR "ZIP")
else()
set(CPACK_GENERATOR "TGZ")
endif()

set(CPACK_PACKAGE_NAME "libnvimgcdcs")
set(CPACK_PACKAGE_VENDOR "NVIDIA CORPORATION")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU Accelerated Image Codecs library")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME})
set(CPACK_STRIP_FILES TRUE)

if(BUILD_TEST)
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_FILE_NAME}_test)
endif()
include (CPack)


# Include custom cmake modules/scripts
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/
)

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

if(BUILD_LIBRARY)
add_subdirectory(src)
endif()

if(BUILD_EXAMPLE)
add_subdirectory(example)
endif()

if(BUILD_PYTHON)
add_subdirectory(python)
endif()

#if(BUILD_DOCS)
#add_subdirectory(docs)
#endif()

if(BUILD_TEST)
add_subdirectory(test)
endif()

set (NVIMGCDCS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)
if(BUILD_EXTENSIONS)
    add_subdirectory(extensions)
endif()