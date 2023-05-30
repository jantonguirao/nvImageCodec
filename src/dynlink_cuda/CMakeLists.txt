set(CUDA_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_cuda_gen.cpp")
add_custom_command(
    OUTPUT ${CUDA_GENERATED_STUB}
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py --unique_prefix=Cuda --
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/cuda.json" ${CUDA_GENERATED_STUB}
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cuda.h" "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"

    # for some reason QNX fails with 'too many errors emitted' is this is not set
    "-ferror-limit=0"
    ${DEFAULT_COMPILER_INCLUDE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cuda.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/cuda.json"
    COMMENT "Running cuda.h stub generator"
    VERBATIM)

set_source_files_properties(${CUDA_GENERATED_STUB} PROPERTIES GENERATED TRUE)
add_library(dynlink_cuda STATIC dynlink_cuda.cpp ${CUDA_GENERATED_STUB})

set_target_properties(dynlink_cuda
    PROPERTIES POSITION_INDEPENDENT_CODE
    ON
)
