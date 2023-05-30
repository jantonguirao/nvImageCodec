if (BUILD_NVJPEG2K_EXT)
    add_subdirectory(nvjpeg2k)
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
