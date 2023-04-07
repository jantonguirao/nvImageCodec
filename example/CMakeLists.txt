add_subdirectory(nvimtrans)
add_subdirectory(example_high_level_api)

if(BUILD_CVCUDA_SAMPLES)
    add_subdirectory(nvimproc)
endif()
