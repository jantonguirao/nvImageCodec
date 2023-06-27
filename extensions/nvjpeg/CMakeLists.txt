set(NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME nvjpeg_ext)

set(NVIMGCDCS_NVJPEG_EXT_SRC
  cuda_decoder.cpp
  hw_decoder.cpp  
  cuda_encoder.cpp
  type_convert.cpp
  nvjpeg_ext.cpp
  nvjpeg_utils.cpp
  exception.cpp    
)

include(CheckCXXSourceCompiles)
  check_cxx_source_compiles(
          "#include <nvjpeg.h>
          int main(){
            return NVJPEG_BACKEND_LOSSLESS_JPEG != 6;
          }"
          NVJPEG_LOSSLESS_SUPPORTED)
  if (NVJPEG_LOSSLESS_SUPPORTED)
    message("nvJPEG lossless supported")
    add_definitions(-DNVJPEG_LOSSLESS_SUPPORTED=1)
  else()
    message("nvJPEG lossless NOT supported")
    add_definitions(-DNVJPEG_LOSSLESS_SUPPORTED=0)
  endif()

if (NVJPEG_LOSSLESS_SUPPORTED)
  list(APPEND NVIMGCDCS_NVJPEG_EXT_SRC lossless_decoder.cpp)
endif()


add_library(${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME} SHARED ${NVIMGCDCS_NVJPEG_EXT_SRC} ext_module.cpp)
add_library(${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME}_static STATIC ${NVIMGCDCS_NVJPEG_EXT_SRC})

if(NOT WITH_DYNAMIC_NVJPEG)
  if (UNIX)
    target_link_libraries(${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME}
      PUBLIC CUDA::cudart_static CUDA::nvjpeg_static CUDA::culibos)
  else()
    target_link_libraries(${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME}
      PUBLIC CUDA::cudart_static CUDA::nvjpeg)
  endif()
else()
  target_link_libraries(${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME}
    PRIVATE CUDA::cudart_static dynlink_nvjpeg)
  target_link_libraries(${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME}
    PRIVATE "-Wl,--exclude-libs,$<TARGET_FILE_NAME:dynlink_nvjpeg>")
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fvisibility=hidden")
  set_target_properties(${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    NO_SONAME OFF)
else()
  set_target_properties(${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    OUTPUT_NAME ${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME}_${PROJECT_VERSION_MAJOR}
    ARCHIVE_OUTPUT_NAME ${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME})
endif()

if(UNIX)
  install(TARGETS ${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME} ${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME}_static
    LIBRARY DESTINATION extensions NAMELINK_SKIP COMPONENT lib
    ARCHIVE DESTINATION lib64 COMPONENT lib
    PUBLIC_HEADER DESTINATION include COMPONENT lib
  )

else()
  install(TARGETS ${NVIMGCDCS_NVJPEG_EXT_LIBRARY_NAME}
    RUNTIME DESTINATION extensions COMPONENT lib
    LIBRARY DESTINATION lib64 COMPONENT lib
    ARCHIVE DESTINATION lib64 COMPONENT lib
    PUBLIC_HEADER DESTINATION include COMPONENT lib
  )
endif()
