


set(SRCS
    nvimgcodecs_tests.cpp
)
# Headers file are located here:
include_directories(
    ../src
    SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(FILESTOPACK nvimgcodecs_tests)

## cmake is unable to find library libcudart.so. work around by adding this
find_library(CUDART_SHARED 
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})


# Find all dependencies
include(Dependencies)

add_executable(nvimgcodecs_tests ${SRCS})

foreach(testapp nvimgcodecs_tests )
    target_include_directories(${testapp} SYSTEM BEFORE
        PRIVATE
        ${GTEST_INCLUDE_DIRS}
    )

    if(UNIX)
        set(PTHREAD pthread)
    endif()

    target_link_libraries(${testapp} PUBLIC nvimgcodecs_static ${CUDART_SHARED} ${GTEST_LIBRARIES} ${PTHREAD})
endforeach()

install(TARGETS ${FILESTOPACK}
    DESTINATION test COMPONENT Applications
)