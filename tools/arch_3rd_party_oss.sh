#!/bin/bash -e

export LIBJPEG_TURBO_VERSION=$(sed -n 's/.*LIBJPEG_TURBO_VERSION\=\([^\ ;]*\).*/\1/p' ../docker/Dockerfile.deps)
export LIBTIFF_VERSION=$(sed -n 's/.*LIBTIFF_VERSION\=\([^\ ;]*\).*/\1/p' ../docker/Dockerfile.deps)
export OPENJPEG_VERSION=$(sed -n 's/.*OPENJPEG_VERSION\=\([^\ ;]*\).*/\1/p' ../docker/Dockerfile.deps)
export OPENCV_VERSION=$(sed -n 's/.*OPENCV_VERSION\=\([^\ ;]*\).*/\1/p' ../docker/Dockerfile.deps)

wget https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/${LIBJPEG_TURBO_VERSION}.tar.gz  -O libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz
wget https://download.osgeo.org/libtiff/tiff-${LIBTIFF_VERSION}.tar.gz -O libtiff-${LIBTIFF_VERSION}.tar.gz
wget https://github.com/uclouvain/openjpeg/archive/refs/tags/v${OPENJPEG_VERSION}.tar.gz -O openjpeg-${OPENJPEG_VERSION}.tar.gz
wget https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz -O opencv-${OPENCV_VERSION}.tar.gz

# submodules
pushd .
cd ..
export DLPACK_SHA=$(git submodule status | grep dlpack | sed s/\ external.*// | sed s/\ //)
export BOOST_PREPROCESSOR_SHA=$(git submodule status | grep boost | sed s/\ external.*// | sed s/\ //)
export GOOGLETEST_SHA=$(git submodule status | grep googletest | sed s/\ external.*// | sed s/\ //)
export PYBIND11_SHA=$(git submodule status | grep pybind11 | sed s/\ external.*// | sed s/\ //)
popd

wget https://github.com/dmlc/dlpack/archive/${DLPACK_SHA}.tar.gz -O dlpack-${DLPACK_SHA}.tar.gz 
wget https://github.com/boostorg/preprocessor/archive/${BOOST_PREPROCESSOR_SHA}.tar.gz -O boost-preprocessor-${BOOST_PREPROCESSOR_SHA}.tar.gz 
wget https://github.com/google/googletest/archive/${GOOGLETEST_SHA}.tar.gz -O googletest-${GOOGLETEST_SHA}.tar.gz
wget https://github.com/pybind/pybind11/archive/${PYBIND11_SHA}.tar.gz -O pybind11-${PYBIND11_SHA}.tar.gz
