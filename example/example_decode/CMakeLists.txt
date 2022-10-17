set(EXAMPLE_NVIMGCDCS_DECODE_NAME example_nvimgcdc_decode)  

set(EXAMPLE_NVIMGCDCS_DECODE_SRC
      main.cpp
)
include_directories(
  ../../include
)   

add_executable(${EXAMPLE_NVIMGCDCS_DECODE_NAME} ${EXAMPLE_NVIMGCDCS_DECODE_SRC})
target_link_libraries(${EXAMPLE_NVIMGCDCS_DECODE_NAME} PUBLIC nvimgcodecs)