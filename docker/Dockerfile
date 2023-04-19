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

# Allow using this image in systems without proper CUDA runtime/driver support.
# We'll be using this image only for building, don't need strict CUDA checks.
ENV NVIDIA_DISABLE_REQUIRE=true

WORKDIR /workspace
