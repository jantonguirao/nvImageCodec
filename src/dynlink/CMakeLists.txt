
DETERMINE_GCC_SYSTEM_INCLUDE_DIRS("c++" "${CMAKE_CXX_COMPILER}" "${CMAKE_CXX_FLAGS}" INFERED_COMPILER_INCLUDE)

# transform a list of paths into a list of include directives
set(DEFAULT_COMPILER_INCLUDE)
foreach(incl_dir ${INFERED_COMPILER_INCLUDE})
  set(DEFAULT_COMPILER_INCLUDE "${DEFAULT_COMPILER_INCLUDE} -I${incl_dir}")
endforeach(incl_dir)
separate_arguments(DEFAULT_COMPILER_INCLUDE UNIX_COMMAND  "${DEFAULT_COMPILER_INCLUDE}")

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

if(WITH_DYNAMIC_NVJPEG)
    set(NVJPEG_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvjpeg_gen.cpp")
    add_custom_command(
        OUTPUT ${NVJPEG_GENERATED_STUB}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py --unique_prefix=Nvjpeg --
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvjpeg.json" ${NVJPEG_GENERATED_STUB}
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/nvjpeg.h" "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"

        # for some reason QNX fails with 'too many errors emitted' is this is not set
        "-ferror-limit=0"
        ${DEFAULT_COMPILER_INCLUDE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/nvjpeg.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvjpeg.json"
        COMMENT "Running nvjpeg.h stub generator"
        VERBATIM)

    set_source_files_properties(${NVJPEG_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_nvjpeg STATIC dynlink_nvjpeg.cpp ${NVJPEG_GENERATED_STUB})
    set_target_properties(dynlink_nvjpeg PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()


if(WITH_DYNAMIC_NVJPEG2K)
    set(NVJPEG2K_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvjpeg2k_gen.cpp")
    add_custom_command(
        OUTPUT ${NVJPEG2K_GENERATED_STUB}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py --unique_prefix=Nvjpeg2k --
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvjpeg2k.json" ${NVJPEG2K_GENERATED_STUB}
        "${NVJPEG2K_INCLUDE}/nvjpeg2k.h" "-I${NVJPEG2K_INCLUDE}" "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"

        # for some reason QNX fails with 'too many errors emitted' is this is not set
        "-ferror-limit=0"
        ${DEFAULT_COMPILER_INCLUDE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py
        "${NVJPEG2K_INCLUDE}/nvjpeg2k.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvjpeg2k.json"
        COMMENT "Running nvjpeg2k.h stub generator"
        VERBATIM)

    set_source_files_properties(${NVJPEG2K_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_nvjpeg2k STATIC dynlink_nvjpeg2k.cpp ${NVJPEG2K_GENERATED_STUB})
    set_target_properties(dynlink_nvjpeg2k PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if (WITH_DYNAMIC_NPP)
    set(NPP_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_npp_gen.cpp")
    add_custom_command(
        OUTPUT ${NPP_GENERATED_STUB}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py --unique_prefix=Npp --
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/npp.json" ${NPP_GENERATED_STUB}
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/npp.h" "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    # for some reason QNX fails with 'too many errors emitted' is this is not set
                    "-ferror-limit=0"
                    ${DEFAULT_COMPILER_INCLUDE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py
                "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/npp.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/npp.json"
        COMMENT "Running npp.h stub generator"
        VERBATIM)

    set_source_files_properties(${NPP_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_npp STATIC dynlink_npp.cpp ${NPP_GENERATED_STUB})
    set_target_properties(dynlink_npp PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
