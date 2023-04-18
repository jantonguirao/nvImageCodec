

set(SRCS
    nvimgcodecs_tests.cpp
    codec_test.cpp
    code_stream_test.cpp
    codec_registry_test.cpp
    plugin_framework_test.cpp
    thread_pool_test.cpp
    processing_results_test.cpp
    device_guard_test.cpp
    parsers/bmp_test.cpp
    parsers/jpeg_test.cpp
    parsers/jpeg2k_test.cpp
    parsers/tiff_test.cpp
    extensions/nvjpeg_ext_parser_test.cpp
    extensions/nvjpeg_ext_encoder_test.cpp
    extensions/nvjpeg2k_ext_encoder_test.cpp
)
# Headers file are located here:
include_directories(
    .
    ..
    ../src
    ../include
    SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(FILESTOPACK nvimgcodecs_tests)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED NVJPEG2K_LIBRARY)
    find_library(NVJPEG2K_LIBRARY NAMES nvjpeg2k)
endif()

# Find all dependencies
include(Dependencies)

add_executable(nvimgcodecs_tests ${SRCS})
foreach(testapp nvimgcodecs_tests)
    target_include_directories(${testapp} SYSTEM BEFORE
        PRIVATE
        ${GTEST_INCLUDE_DIRS}
    )

    if(UNIX)
        set(PTHREAD pthread)
    endif()

    target_link_libraries(${testapp} PUBLIC
        nvimgcodecs_static
        nvjpeg_ext_static
        nvjpeg2k_ext_static
        ${NVJPEG2K_LIBRARY}
        CUDA::nvjpeg
        CUDA::cudart
        CUDA::cuda_driver
        GTest::gtest
        GTest::gmock
        ${PTHREAD})
endforeach()

install(TARGETS ${FILESTOPACK}
    DESTINATION test COMPONENT Applications
)

add_test(NAME nvImageCodecsTest-L0 COMMAND nvimgcodecs_tests --gtest_filter=-*Parser*:*Ext* --resources_dir ../resources WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME nvImageCodecsTest-L1-Builtin_Parsers COMMAND nvimgcodecs_tests --gtest_filter=*Parser* --resources_dir ../resources WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME nvImageCodecsTest-L1-Extensions COMMAND nvimgcodecs_tests --gtest_filter=*Ext* --resources_dir ../resources WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#TODO add_test(NAME nvImageCodecsTest-L2-Transcoding COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
