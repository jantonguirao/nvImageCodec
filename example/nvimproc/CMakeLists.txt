set(NVIMGCODEC_EXAMPLE_NAME nvimproc)

set(NVIMGCODEC_EXAMPLE_SRC
        main.cpp
)

find_package(NVCV_TYPES REQUIRED)
find_package(CVCUDA REQUIRED)

add_executable(${NVIMGCODEC_EXAMPLE_NAME} ${NVIMGCODEC_EXAMPLE_SRC})

target_link_libraries(${NVIMGCODEC_EXAMPLE_NAME} PUBLIC CUDA::cudart nvimgcodec nvcv_types cvcuda)

install(TARGETS ${NVIMGCODEC_EXAMPLE_NAME}
        DESTINATION bin COMPONENT lib
)