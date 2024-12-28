#include "vec.cu"

template <typename T>
class Vec {
    private: 
        size_t size; 
        // points to a location on device NOT host
        T* data_ptr; 

    public:
        // initialize vector of zeros 
        Vec(size_t size) : size(size), data_ptr(nullptr) {
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

        // initialize from vector of values 
        Vec(std::vector<T> vec) : size(vec.size()), data_ptr(nullptr) {
            cudaError_t err = cudaMalloc(&data_ptr, size * sizeof(T));
            if (err != cudaSuccess) {
                std::cerr << "Vec malloc failed: " << cudaGetErrorString(err) << std::endl;
                std::terminate(); 
            }
            cudaMemcpy(data_ptr, vec.data(), size * sizeof(T), cudaMemcpyHostToDevice);
        }

        // destructor for device memory 
        ~Vec() {
            cudaError_t err = cudaFree(d_data);
            if (err != cudaSuccess) {
                std::cerr << "Vec destructor failed to delete device pointer: " << cudaGetErrorString(err) << std::endl;
            }
        }

        T norm() {
            return (T) 0; 
        }

        // element-wise operators 
        friend Vec<T> operator+(const Vec<T>& A, const Vec<T>& B);
        friend Vec<T> operator*(const Vec<T>& A, const Vec<T>& B);
        friend Vec<T> operator/(const Vec<T>& A, const Vec<T>& B);
        friend Vec<T> operator-(const Vec<T>& A, const Vec<T>& B);

        // dot product [TODO: Add an exterior product]
        friend Vec<T> operator%(const Vec<T>& A, const Vec<T>& B);
}