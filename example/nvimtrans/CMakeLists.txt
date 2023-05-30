set(NVIMGCDCS_EXAMPLE_NAME nvimtrans)  

set(NVIMGCDCS_EXAMPLE_SRC
      main.cpp
)
include_directories(
  ../../
  ../../include
)   

find_library(CUDART_SHARED 
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    )


add_executable(${NVIMGCDCS_EXAMPLE_NAME} ${NVIMGCDCS_EXAMPLE_SRC})

set_property(TARGET ${NVIMGCDCS_EXAMPLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${NVIMGCDCS_EXAMPLE_NAME} PUBLIC nvimgcodecs nvpnm_ext_static CUDA::cudart)

install(TARGETS ${NVIMGCDCS_EXAMPLE_NAME}
    DESTINATION bin COMPONENT Applications
  )
