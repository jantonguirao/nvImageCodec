set(NVIMGCDCS_LIBRARY_NAME nvimgcodecs)  

set (NVIMGCDCS_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodecs.h
    nvimgcdcs_capi.cpp
    exception.cpp
    plugin_framework.cpp
    codec_registry.cpp
    codec.cpp
    image_decoder.cpp
    image_encoder.cpp
    image_parser.cpp
    code_stream.cpp
    file_io_stream.cpp
    std_file_io_stream.cpp
    decode_state.cpp
    image.cpp
    parse_state.cpp
    encode_state.cpp
)

include_directories(../include)  

# Build the library
message(STATUS "props ProductVersionFiles ${ProductVersionFiles}")  

add_library(${NVIMGCDCS_LIBRARY_NAME} SHARED ${NVIMGCDCS_SRCS} ${ProductVersionFiles})
if(UNIX)
    target_link_libraries (${NVIMGCDCS_LIBRARY_NAME} PRIVATE dl)
endif()

# Configure library version
include(ConfigureVersion)
configure_library_version(NVIMGCDCS_LIBRARY_NAME
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcdcs_version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcdcs_version.h"
)

