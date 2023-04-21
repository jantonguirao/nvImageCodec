set(NVIMGCDCS_LIBRARY_NAME nvimgcodecs)

set(NVIMGCDCS_SRCS
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
    image.cpp
    parse_state.cpp
    logger.cpp
    default_debug_messenger.cpp
    image_parser_factory.cpp
    image_encoder_factory.cpp
    image_decoder_factory.cpp
    image_generic_decoder.cpp
    image_generic_encoder.cpp
    nvimgcodecs_director.cpp
    processing_results.cpp
    decode_state_batch.cpp
    encode_state_batch.cpp
    thread_pool.cpp
    device_guard.cpp
    default_executor.cpp
    parsers/bmp.cpp
    parsers/exif.cpp
    parsers/jpeg.cpp
    parsers/jpeg2k.cpp
    parsers/tiff.cpp
    parsers/png.cpp
    parsers/pnm.cpp
    parsers/webp.cpp
)

include_directories(
    .
    ../include
    SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Build the library
message(STATUS "props ProductVersionFiles ${ProductVersionFiles}")

add_library(${NVIMGCDCS_LIBRARY_NAME} SHARED ${NVIMGCDCS_SRCS} ${ProductVersionFiles})
add_library(${NVIMGCDCS_LIBRARY_NAME}_static STATIC ${NVIMGCDCS_SRCS} ${ProductVersionFiles})

set(LIBTOPACK ${NVIMGCDCS_LIBRARY_NAME} ${NVIMGCDCS_LIBRARY_NAME}_static)

find_library(CUDART_SHARED
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
)

find_library(CUDA_LIB
    NAMES cuda
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
)

target_link_libraries(${NVIMGCDCS_LIBRARY_NAME} PUBLIC ${CUDART_SHARED} ${CUDA_LIB})

if(UNIX)
    # CXX flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fvisibility=hidden")
    target_link_libraries(${NVIMGCDCS_LIBRARY_NAME} PRIVATE dl)
    target_link_libraries(${NVIMGCDCS_LIBRARY_NAME}_static PRIVATE dl)
endif()


# Configure library version
include(ConfigureVersion)
configure_library_version(NVIMGCDCS_LIBRARY_NAME
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcdcs_version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcdcs_version.h"
)

set(NVIMGCDCS_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodecs.h")
set(NVIMGCDCS_PUBLIC_HEADERS "${NVIMGCDCS_PUBLIC_HEADERS}\;${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcdcs_version.h")
set(NVIMGCDCS_PUBLIC_HEADERS "${NVIMGCDCS_PUBLIC_HEADERS}\;${CMAKE_CURRENT_SOURCE_DIR}/../include/nvcv_adapter.hpp")

set_target_properties(${NVIMGCDCS_LIBRARY_NAME} PROPERTIES PUBLIC_HEADER ${NVIMGCDCS_PUBLIC_HEADERS})

install(TARGETS
    ${LIBTOPACK}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib64
    ARCHIVE DESTINATION lib64
    PUBLIC_HEADER DESTINATION include
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.txt
    DESTINATION .
)

if(UNIX)
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/install.sh
        DESTINATION .
    )
endif()
