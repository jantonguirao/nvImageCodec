add_subdirectory(nvjpeg2k)
find_library(NVJPEG2K_LIBRARY NAMES nvjpeg2k)

if(${NVJPEG2K_LIBRARY} STREQUAL "NVJPEG2K_LIBRARY-NOTFOUND")
    message(WARNING "nvjpeg2000 not found - disabled. Use following link to download https://developer.nvidia.com/nvjpeg2000/downloads")
    set(BUILD_NVJPEG2K_EXT OFF CACHE BOOL INTERNAL)
    set(BUILD_NVJPEG2K_EXT OFF)
else()
    message(NOTICE "Found nvjpeg2000: " ${NVJPEG2K_LIBRARY})
endif()

if(BUILD_NVJPEG2K_EXT)
    add_subdirectory(nvjpeg2k)
endif ()

find_library(NVJPEG_LIBRARY NAMES nvjpeg PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

if(${NVJPEG_LIBRARY} STREQUAL "NVJPEG_LIBRARY-NOTFOUND")
    message(WARNING "nvjpeg not found - disabled.")
    set(BUILD_NVJPEG_EXT OFF CACHE BOOL INTERNAL)
    set(BUILD_NVJPEG_EXT OFF)
else()
    message(NOTICE "Found nvjpeg: " ${NVJPEG_LIBRARY})
    set(BUILD_NVJPEG_EXT ON)
endif()

if(BUILD_NVJPEG_EXT)
    add_subdirectory(nvjpeg)
endif()

