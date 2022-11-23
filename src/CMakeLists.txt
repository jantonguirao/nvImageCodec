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

include_directories(
    ../include
    SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})  

message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Build the library

message(STATUS "props ProductVersionFiles ${ProductVersionFiles}")  

add_library(${NVIMGCDCS_LIBRARY_NAME} SHARED ${NVIMGCDCS_SRCS} ${ProductVersionFiles})
add_library(${NVIMGCDCS_LIBRARY_NAME}_static STATIC   ${NVIMGCDCS_SRCS} ${ProductVersionFiles})

set(LIBTOPACK ${NVIMGCDCS_LIBRARY_NAME} ${NVIMGCDCS_LIBRARY_NAME}_static)

find_library(CUDART_SHARED 
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    )
target_link_libraries(${NVIMGCDCS_LIBRARY_NAME} PUBLIC ${CUDART_SHARED})

if(UNIX)
    # CXX flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fvisibility=hidden")
    target_link_libraries (${NVIMGCDCS_LIBRARY_NAME} PRIVATE dl)
endif()

# Configure library version
include(ConfigureVersion)
configure_library_version(NVIMGCDCS_LIBRARY_NAME
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcdcs_version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcdcs_version.h"
)

set(NVIMGCDCS_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodecs.h")
set(NVIMGCDCS_PUBLIC_HEADERS "${NVIMGCDCS_PUBLIC_HEADERS}\;${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcdcs_module.h")
set(NVIMGCDCS_PUBLIC_HEADERS "${NVIMGCDCS_PUBLIC_HEADERS}\;${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcdcs_version.h")

set_target_properties(${NVIMGCDCS_LIBRARY_NAME} PROPERTIES PUBLIC_HEADER ${NVIMGCDCS_PUBLIC_HEADERS})

install(TARGETS
${LIBTOPACK} DESTINATION lib64
PUBLIC_HEADER DESTINATION include
)

install(FILES
${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.txt
DESTINATION .
)
