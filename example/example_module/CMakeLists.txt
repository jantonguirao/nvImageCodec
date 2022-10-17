set(EXAMPLE_NVIMGCDCS_LIBRARY_NAME example_nvimgcdc_module)  

set(EXAMPLE_NVIMGCDCS_SRC
      example_module.cpp
      example_encoder.cpp
      example_decoder.cpp
)
include_directories(
  ../../include
)     
add_library(${EXAMPLE_NVIMGCDCS_LIBRARY_NAME} MODULE ${EXAMPLE_NVIMGCDCS_SRC})