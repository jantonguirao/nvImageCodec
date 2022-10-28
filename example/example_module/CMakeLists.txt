set(EXAMPLE_NVIMGCDCS_LIBRARY_NAME example_nvimgcdc_module)  

set(EXAMPLE_NVIMGCDCS_SRC
      example_module.cpp
      example_encoder.cpp
      example_decoder.cpp
      example_parser.cpp
)
include_directories(
  ../../include
  SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)     

find_library(CUDART_SHARED 
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    )
    
add_library(${EXAMPLE_NVIMGCDCS_LIBRARY_NAME} MODULE ${EXAMPLE_NVIMGCDCS_SRC})
target_link_libraries(${EXAMPLE_NVIMGCDCS_LIBRARY_NAME} PUBLIC ${CUDART_SHARED})