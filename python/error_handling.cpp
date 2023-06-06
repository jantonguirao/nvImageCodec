
#include "error_handling.h"
#include <stdexcept>
#include <cuda_runtime.h>

void check_cuda_buffer(const void* ptr)
{
    if (ptr == nullptr) {
        throw std::runtime_error("NULL CUDA buffer not accepted");
    }

    cudaPointerAttributes attrs = {};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    cudaGetLastError(); // reset the cuda error (if any)
    if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered) {
        throw std::runtime_error("Buffer is not CUDA-accessible");
    }
}