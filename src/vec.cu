#include "vec.hpp"
#include <stdexcept>
#include <iostream> 

constexpr int BLOCK_SIZE = 512;

// Absolute
template <typename T>
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
        sdata[thread_index] += pow(abs(vec[idx]), p);
        idx += blockDim.x * gridDim.x;   
    }

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (thread_index < s) sdata[thread_index] += pow(abs(sdata[thread_index + s]), p);
    }
    if (thread_index == 0) atomicAdd(out, pow(sdata[0], (T) 1.0 / p));
}

// dot product 
template <typename T>
__global__ void vec_dot(const T* vecA, const T* vecB, int size, T* out) {
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

template <typename T> 
T operator%(const Vec<T>& A, const Vec<T>& B) {
    if (A.size != B.size) {
        throw std::invalid_argument("Vectors are not the same size.");
    }

    T* out_device; 
    cudaMalloc(&out_device, sizeof(T));
    vec_dot<<<100,1024>>>(A.data_ptr, B.data_ptr, A.size, out_device);

    cudaDeviceSynchronize();

    T out_host; 
    cudaMemcpy(&out_host, out_device, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(out_device);

    return out_host; 
}

template <typename T>
Vec<T>::~Vec() {
    cudaError_t err = cudaFree(data_ptr);
    if (err != cudaSuccess) {
        std::cerr << "Vec destructor failed to delete device pointer: " << cudaGetErrorString(err) << std::endl;
    }
}

template <typename T>
Vec<T>::Vec(size_t size) : size(size), data_ptr(nullptr) {
    cudaError_t err = cudaMalloc(&data_ptr, size * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "Vec malloc failed: " << cudaGetErrorString(err) << std::endl;
        std::terminate(); 
    }

    cudaError_t memset_err = cudaMemset(data_ptr, 0, size * sizeof(T));
    if (memset_err != cudaSuccess) {
        std::cerr << "Setting memory to zeros failed: " << cudaGetErrorString(err) << std::endl;
        std::terminate();
    }
}

template <typename T>
Vec<T>::Vec(const std::vector<T>& vec) : size(vec.size()), data_ptr(nullptr) {
    cudaError_t err = cudaMalloc(&data_ptr, size * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "Vec malloc failed: " << cudaGetErrorString(err) << std::endl;
        std::terminate(); 
    }
    cudaMemcpy(data_ptr, vec.data(), size * sizeof(T), cudaMemcpyHostToDevice);
}

template class Vec<float>;
template class Vec<double>;
template float operator%<float>(const Vec<float>&, const Vec<float>&);
template double operator%<double>(const Vec<double>&, const Vec<double>&);