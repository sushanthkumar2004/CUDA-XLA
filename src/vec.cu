#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>

#define BLOCK_SIZE 512

// Absolute
__global__ void vec_abs(const T* vecA, const T* vecOut, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vecOut[idx] = abs(vecA[idx]);
    }
}

// Add two vectors element-wise
template <typename T>
__global__ void vec_add(const T* vecA, const T* vecB, T* vecOut, int size) {
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;            

    for (int i = idx; i < size; i += stride) {
        vecOut[i] = vecA[i] + vecB[i];
    }
}

// Multiply two vectors element-wise
template <typename T>
__global__ void vec_mult(const T* vecA, const T* vecB, T* vecOut, int size) {
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;            

    for (int i = idx; i < size; i += stride) {
        vecOut[i] = vecA[i] * vecB[i];
    }
}

// Divide two vectors element-wise
template <typename T>
__global__ void vec_div(const T* vecA, const T* vecB, T* vecOut, int size) {
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;            

    for (int i = idx; i < size; i += stride) {
        vecOut[i] = vecA[i] / vecB[i];
    }
}

// Subtract two vectors element-wise
template <typename T>
__global__ void vec_sub(const T* vecA, const T* vecB, T* vecOut, int size) {
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;            

    for (int i = idx; i < size; i += stride) {
        vecOut[i] = vecA[i] - vecB[i];
    }
}

// Scale by scalar 
template <typename T>
__global__ void vec_scale(T* vecA, int size, T lambda) {
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;            

    for (int i = idx; i < size; i += stride) {
        vecA[i] *= lambda; 
    }
}

// Lp norm 
template <typename T>
__global__ void vec_lp(const T* vec, int size, T p, T* out) {
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");

    __shared__ T sdata[BLOCK_SIZE];

    int thread_index = threadIdx.x;
    sdata[threadIdx.x] = 0;
    size_t idx = threadIdx.x + blockDim.x*blockIdx.x;

    while (idx < size) {
        sdata[thread_index] += pow(gdata[idx], p);
        idx += blockDim.x * gridDim.x;   
    }

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (thread_index < s) sdata[thread_index] += pow(sdata[thread_index + s], p);
    }
    if (thread_index == 0) atomicAdd(out, pow(sdata[0], (T) 1.0 / p));
}

// dot product 
template <typename T>
__global__ void vec_dot(const T* vecA, const T* vecB, int size, T p) {
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");

    __shared__ T sdata[BLOCK_SIZE];

    int thread_index = threadIdx.x;
    sdata[threadIdx.x] = 0;
    size_t idx = threadIdx.x + blockDim.x*blockIdx.x;

    while (idx < size) {
        sdata[thread_index] += vecA[idx] * vecB[idx]; 
        idx += blockDim.x * gridDim.x;   
    }

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (thread_index < s) sdata[thread_index] += vecA[thread_index + s] * vecB[thread_index + s];
    }
    if (thread_index == 0) atomicAdd(out, sdata[0]);
}
