
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
    list(APPEND SRCS extensions/nvjpeg_ext_encoder_test.cpp)
    list(APPEND SRCS extensions/nvjpeg_ext_decoder_test.cpp)
endif()


if (BUILD_NVJPEG2K_EXT)
    list(APPEND SRCS extensions/nvjpeg2k_ext_encoder_test.cpp)
    list(APPEND SRCS extensions/nvjpeg2k_ext_decoder_test.cpp)
endif()

if (BUILD_LIBJPEG_TURBO_EXT)
    list(APPEND SRCS extensions/libjpeg_turbo_ext_decoder_test.cpp)
endif()

if (BUILD_LIBTIFF_EXT)
    list(APPEND SRCS extensions/libtiff_ext_decoder_test.cpp)
endif()

if (BUILD_OPENCV_EXT)
    list(APPEND SRCS extensions/opencv_ext_decoder_test.cpp)
endif()

set(FILESTOPACK nvimgcodecs_tests)

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
        #TODO On Windows linking with OpenCV_LIBRARIES already takes tiff library
        if (NOT WIN32)
        list(APPEND TARGET_LIBS ${TIFF_LIBRARY})
        endif()
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
        CUDA::nppc
        CUDA::nppidei
        CUDA::cuda_driver
        gtest
        gmock
        ${PTHREAD})
endforeach()

install(TARGETS ${FILESTOPACK}
    DESTINATION test COMPONENT Applications
)

add_test(NAME nvImageCodecsTest-L0 COMMAND nvimgcodecs_tests --gtest_filter=-*Parser*:*Ext* --resources_dir ../resources WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME nvImageCodecsTest-L1-Builtin_Parsers COMMAND nvimgcodecs_tests --gtest_filter=*Parser* --resources_dir ../resources WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME nvImageCodecsTest-L1-Extensions COMMAND nvimgcodecs_tests --gtest_filter=*Ext* --resources_dir ../resources WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#TODO add_test(NAME nvImageCodecsTest-L2-Transcoding COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
