# CMake script for downloading, unpacking and building dependency at configure time
include(third_party/DownloadProject)



# ###############################################################
# Google Test and OpenJPEG
# ###############################################################
if(BUILD_TEST)
        find_package(GTest QUIET)
        if(NOT GTEST_FOUND)
            find_package(Git REQUIRED)
            if(NOT Git_FOUND)
                message(FATAL_ERROR "Git not installed")
            endif()

            message(STATUS "Building Google Test")
            set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
            download_project(PROJ                googletest
                            GIT_REPOSITORY      https://github.com/google/googletest.git
                            GIT_TAG             release-1.10.0
                            INSTALL_DIR         ${GTEST_ROOT}
                            CMAKE_ARGS          -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_GMOCK=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                            LOG_DOWNLOAD        TRUE
                            LOG_CONFIGURE       TRUE
                            LOG_BUILD           TRUE
                            LOG_INSTALL         TRUE
                            UPDATE_DISCONNECTED TRUE
            )
        endif()
        find_package(GTest REQUIRED)

        find_package(OpenJPEG QUIET)
        if(NOT OPENJPEG_FOUND)
            find_package(Git REQUIRED)
            if(NOT Git_FOUND)
                message(FATAL_ERROR "Git not installed")
            endif()

            message(STATUS "Building openjpeg")
            set(OPENJPEG_ROOT ${CMAKE_CURRENT_BINARY_DIR}/openjpeg CACHE PATH "")
            download_project(PROJ               openjpeg
                            GIT_REPOSITORY      https://github.com/uclouvain/openjpeg.git
                            GIT_TAG             v2.3.1
                            INSTALL_DIR         ${OPENJPEG_ROOT}
                            CMAKE_ARGS          -DBUILD_CODEC=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                            LOG_DOWNLOAD        TRUE
                            LOG_CONFIGURE       TRUE
                            LOG_BUILD           TRUE
                            LOG_INSTALL         TRUE
                            UPDATE_DISCONNECTED TRUE
                          )
        endif()
        set(CMAKE_PREFIX_PATH ${OPENJPEG_ROOT}/lib/openjpeg-2.3)
        find_package(OpenJPEG REQUIRED)



endif()
