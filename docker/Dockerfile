ARG VER_CUDA=11.8.0
ARG VER_UBUNTU=20.04

FROM nvidia/cuda:$VER_CUDA-devel-ubuntu$VER_UBUNTU

ARG DEBIAN_FRONTEND=noninteractive

RUN  apt-get update \
    && apt-get install -y --no-install-recommends \
        git git-lfs \
        g++-11 \
        cmake \
        curl \
        scons \
	wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /opt/cmake-3.24.1 \
      && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
      && rm /tmp/cmake-install.sh \
      && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3 \
	 python3-dev \
         python3-pytest \
         python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install numba \
    && rm -rf /root/.cache/pip

RUN  apt-get update \
    && apt-get -y install nvjpeg2k

RUN export LIBJPEG_TURBO_VERSION=2.1.91 && cd /tmp && \
    wget https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/${LIBJPEG_TURBO_VERSION}.tar.gz && \
    tar xzvf ${LIBJPEG_TURBO_VERSION}.tar.gz && cd libjpeg-turbo-${LIBJPEG_TURBO_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=/usr/local -DENABLE_SHARED=FALSE && \
    make -j8 && make install && \
    cd && rm -rf /tmp/${LIBJPEG_TURBO_VERSION}.tar.gz /tmp/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}

RUN export OPENCV_VERSION=4.7.0 && cd /tmp && \
    wget https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz && \
    tar xzvf ${OPENCV_VERSION}.tar.gz && cd opencv-${OPENCV_VERSION} && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
      -DVIBRANTE_PDK:STRING=/ \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_LIST=core,improc,imgcodecs \
      -DBUILD_SHARED_LIBS=OFF \
      -DWITH_ADE=OFF \
      -DWITH_CUDA=OFF \
      -DWITH_1394=OFF \
      -DWITH_IPP=OFF \
      -DWITH_OPENCL=OFF \
      -DWITH_GTK=OFF \
      -DBUILD_JPEG=OFF \
      -DWITH_JPEG=ON \
      -DBUILD_TIFF=ON \
      -DWITH_TIFF=OFF \
      -DBUILD_JASPER=OFF \
      -DBUILD_DOCS=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_PNG=ON \
      -DWITH_WEBP=ON \
      -DBUILD_opencv_cudalegacy=OFF \
      -DBUILD_opencv_stitching=OFF \
      -DWITH_TBB=OFF \
      -DWITH_QUIRC=OFF \
      -DWITH_OPENMP=OFF \
      -DWITH_PTHREADS_PF=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_java=OFF \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_opencv_python3=OFF \
      -DWITH_PROTOBUF=OFF \
      -DWITH_FFMPEG=OFF \
      -DWITH_GSTREAMER=OFF \
      -DWITH_GSTREAMER_0_10=OFF \
      -DWITH_VTK=OFF \
      -DWITH_OPENEXR=OFF \
      -DINSTALL_C_EXAMPLES=OFF \
      -DINSTALL_TESTS=OFF \
      -DVIBRANTE=TRUE \
      -DWITH_CSTRIPES=OFF \
      -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
      VERBOSE=1 .. && \
      make -j8 && make install && \
      cd && rm -rf /tmp/${OPENCV_VERSION}.tar.gz /tmp/opencv-${OPENCV_VERSION}

# Allow using this image in systems2.1.91.tar.gz without proper CUDA runtime/driver support.
# We'll be using this image only for building, don't need strict CUDA checks.
ENV NVIDIA_DISABLE_REQUIRE=true

WORKDIR /workspace
