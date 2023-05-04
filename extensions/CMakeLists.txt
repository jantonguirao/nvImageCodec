if(NOT DEFINED NVJPEG2K_LIBRARY)
    find_library(NVJPEG2K_LIBRARY NAMES nvjpeg2k)
endif()

if(${NVJPEG2K_LIBRARY} STREQUAL "NVJPEG2K_LIBRARY-NOTFOUND")
    message(WARNING "nvjpeg2000 not found - disabled. Use following link to download https://developer.nvidia.com/nvjpeg2000/downloads")
    set(BUILD_NVJPEG2K_EXT OFF CACHE BOOL INTERNAL)
    set(BUILD_NVJPEG2K_EXT OFF)
else()
    message(NOTICE "Found nvjpeg2000: " ${NVJPEG2K_LIBRARY})
endif()

if(BUILD_NVJPEG2K_EXT)
    add_subdirectory(nvjpeg2k)
endif()

if(NOT DEFINED NVJPEG_LIBRARY)
    find_library(NVJPEG_LIBRARY NAMES nvjpeg PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

if(${NVJPEG_LIBRARY} STREQUAL "NVJPEG_LIBRARY-NOTFOUND")
    message(WARNING "nvjpeg not found - disabled.")
    set(BUILD_NVJPEG_EXT OFF CACHE BOOL INTERNAL)
    set(BUILD_NVJPEG_EXT OFF)
else()
    message(NOTICE "Found nvjpeg: " ${NVJPEG_LIBRARY})
endif()

if(BUILD_NVJPEG_EXT)
    add_subdirectory(nvjpeg)
endif()

if(BUILD_NVBMP_EXT)
    add_subdirectory(nvbmp)
endif()

if(BUILD_NVPNM_EXT)
    add_subdirectory(nvpnm)
endif()

if(BUILD_LIBJPEG_TURBO_EXT)
    add_subdirectory(libjpeg_turbo)
endif ()

if(BUILD_LIBTIFF_EXT)
    add_subdirectory(libtiff)
endif ()

if(BUILD_OPENCV_EXT)
    add_subdirectory(opencv)
endif ()
