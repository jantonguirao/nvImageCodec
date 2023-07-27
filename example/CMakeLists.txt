add_subdirectory(nvimtrans)

if(BUILD_CVCUDA_SAMPLES)
    add_subdirectory(nvimproc)
endif()

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/nvimtrans
    ${CMAKE_CURRENT_SOURCE_DIR}/nvimproc
    ${CMAKE_CURRENT_SOURCE_DIR}/python
    DESTINATION samples
    COMPONENT samples
)
