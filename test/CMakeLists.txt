

set(SRCS
    nvimgcodecs_tests.cpp
    test_utils.cpp
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
    parsers/png_test.cpp
    parsers/pnm_test.cpp
    parsers/webp_test.cpp
    api/can_decode_test.cpp
    api/can_encode_test.cpp
)

if (BUILD_NVJPEG_EXT)
    list(APPEND SRCS extensions/nvjpeg_ext_parser_test.cpp)
    list(APPEND SRCS extensions/nvjpeg_ext_encoder_test.cpp)
    list(APPEND SRCS extensions/nvjpeg_ext_decoder_test.cpp)
endif()


if (BUILD_NVJPEG2K_EXT)
    if(NOT DEFINED NVJPEG2K_LIBRARY)
        find_library(NVJPEG2K_LIBRARY NAMES nvjpeg2k)
    endif()

    list(APPEND SRCS extensions/nvjpeg2k_ext_encoder_test.cpp)
    list(APPEND SRCS extensions/nvjpeg2k_ext_decoder_test.cpp)
endif()

if (BUILD_LIBJPEG_TURBO_EXT)
    if(NOT DEFINED JPEG_LIBRARY)
        find_package(JPEG 62 REQUIRED) # 1.5.3 version
        message("Using libjpeg-turbo at ${JPEG_LIBRARY}")
    endif()

    list(APPEND SRCS extensions/libjpeg_turbo_ext_decoder_test.cpp)
endif()

if (BUILD_LIBTIFF_EXT)
    if(NOT DEFINED TIFF_LIBRARY)
        find_package(TIFF REQUIRED)
        message("Using libtiff at ${TIFF_LIBRARY}")
    endif()
    list(APPEND SRCS extensions/libtiff_ext_decoder_test.cpp)
endif()

if (BUILD_OPENCV_EXT)
    list(APPEND SRCS extensions/opencv_ext_decoder_test.cpp)
endif()

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

if (NOT DEFINED OpenCV_VERSION)
  find_package(OpenCV 4.0 QUIET COMPONENTS core imgproc imgcodecs)
  if(NOT OpenCV_FOUND)
    find_package(OpenCV 3.0 REQUIRED COMPONENTS core imgproc imgcodecs)
  endif()

  message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} (found suitable version \"${OpenCV_VERSION}\", minimum required is \"3.0\")")
  message("OpenCV libraries: ${OpenCV_LIBRARIES}")
endif()


include_directories(
  ../../include
  SYSTEM ${JPEG_INCLUDE_DIR}
  SYSTEM ${OpenCV_INCLUDE_DIRS}
  SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

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

    set(TARGET_LIBS nvimgcodecs_static)

    if (BUILD_NVJPEG_EXT)
        list(APPEND TARGET_LIBS nvjpeg_ext_static)
    endif()

    if (BUILD_NVJPEG2K_EXT)
        list(APPEND TARGET_LIBS nvjpeg2k_ext_static)
    endif()

    if (BUILD_LIBJPEG_TURBO_EXT)
        list(APPEND TARGET_LIBS libjpeg_turbo_ext_static)
        list(APPEND TARGET_LIBS ${JPEG_LIBRARY})
        
    endif()

    if (BUILD_LIBTIFF_EXT)
        list(APPEND TARGET_LIBS libtiff_ext_static)
        list(APPEND TARGET_LIBS ${TIFF_LIBRARY})
        list(APPEND TARGET_LIBS ${JPEG_LIBRARY})
    endif()

    if (BUILD_OPENCV_EXT)
        list(APPEND TARGET_LIBS opencv_ext_static)
    endif()

    target_link_libraries(${testapp} PUBLIC
        ${TARGET_LIBS}
        ${OpenCV_LIBRARIES}
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
