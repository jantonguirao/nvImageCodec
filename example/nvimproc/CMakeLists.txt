set(NVIMGCDCS_EXAMPLE_NAME nvimproc)

set(NVIMGCDCS_EXAMPLE_SRC
        main.cpp
)

include_directories(
        ../../include
)

include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(CUDAToolkit REQUIRED)
find_package(NVCV_TYPES REQUIRED)
find_package(CVCUDA REQUIRED)

add_executable(${NVIMGCDCS_EXAMPLE_NAME} ${NVIMGCDCS_EXAMPLE_SRC})

target_link_libraries(${NVIMGCDCS_EXAMPLE_NAME} PUBLIC nvimgcodecs nvcv_types cvcuda)

install(TARGETS ${NVIMGCDCS_EXAMPLE_NAME}
        DESTINATION bin COMPONENT Applications
)