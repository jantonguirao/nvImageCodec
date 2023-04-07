


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
    )
# Headers file are located here:
include_directories(
    .
    ../src
    ../include
    SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(FILESTOPACK nvimgcodecs_tests)

## cmake is unable to find library libcudart.so. work around by adding this
find_library(CUDART_SHARED 
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

find_library(CUDA_LIB
    NAMES cuda
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
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

    target_link_libraries(${testapp} PUBLIC nvimgcodecs_static ${CUDART_SHARED} ${CUDA_LIB} GTest::gtest GTest::gmock ${PTHREAD})
endforeach()

install(TARGETS ${FILESTOPACK}
    DESTINATION test COMPONENT Applications
)

add_test(NAME nvImageCodecsTest-L0 COMMAND nvimgcodecs_tests)
add_test(NAME nvImageCodecsTest-L2 COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
