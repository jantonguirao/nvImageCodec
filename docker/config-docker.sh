export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
export REGISTRY_PREFIX=${REGISTRY_PREFIX:-"gitlab-master.nvidia.com:5005/cuda-hpc-libraries/nvimagecodec/"}
export PLATFORM=${PLATFORM:-"linux/amd64"}  # or "linux/arm64"
export ARCH=${ARCH:-"x86_64"}  # or "aarch64"