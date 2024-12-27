#include<vector>



<template typename T> 
class Vec {
    private: 
        size_t size; 
        // points to a location on device 
        T* data_ptr; 
    
    public:
        Vec(size_t size) : size(size), data_ptr(nullptr) {
            cudaError_t err = cudaMalloc(&data_ptr, size * sizeof(T));
            if (err != cudaSuccess) {
                std::cerr << "Vec malloc failed: " << cudaGetErrorString(err) << std::endl;
                std::terminate(); 
            }
        }

        Vec(std::vector<T> vec) : size(vec.size()), data_ptr(nullptr) {
            cudaError_t err = cudaMalloc(&data_ptr, size * sizeof(T));
            if (err != cudaSuccess) {
                std::cerr << "Vec malloc failed: " << cudaGetErrorString(err) << std::endl;
                std::terminate(); 
            }


        }

        ~Vec() {
            cudaError_t err = cudaFree(d_data);
            if (err != cudaSuccess) {
                std::cerr << "Vec destructor failed to delete device pointer: " << cudaGetErrorString(err) << std::endl;
            }
        }

}