#include "matrix.cu"

template <typename T>
class Matrix {
    private: 
        size_t rows; 
        size_t cols; 
        // points to a location on device NOT host 
        // the data is row ordered. 
        T* data_ptr; 

    public:
        // initialize vector of zeros 
        Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data_ptr(nullptr) {
            cudaError_t err = cudaMalloc(&data_ptr, rows * cols * sizeof(T));
            if (err != cudaSuccess) {
                std::cerr << "Matrix malloc failed: " << cudaGetErrorString(err) << std::endl;
                std::terminate(); 
            }

            cudaError_t memset_err = cudaMemset(data_ptr, 0, rows * cols * sizeof(T));
            if (memset_err != cudaSuccess) {
                std::cerr << "Setting memory to zeros failed: " << cudaGetErrorString(err) << std::endl;
                std::terminate();
            }
        }

        // initialize from array of values 
        Matrix(std::vector<vector<T>> matrix) : rows(matrix.size()), cols(matrix[0].size()), data_ptr(nullptr) {
            cudaError_t err = cudaMalloc(&data_ptr, rows * cols * sizeof(T));
            if (err != cudaSuccess) {
                std::cerr << "Vec malloc failed: " << cudaGetErrorString(err) << std::endl;
                std::terminate(); 
            }
            cudaMemcpy(data_ptr, vec.data(), rows * cols * sizeof(T), cudaMemcpyHostToDevice);
        }

        // destructor for device memory 
        ~Matrix() {
            cudaError_t err = cudaFree(d_data);
            if (err != cudaSuccess) {
                std::cerr << "Vec destructor failed to delete device pointer: " << cudaGetErrorString(err) << std::endl;
            }
        }

        T norm() {
            return (T) 0; 
        }

        // element-wise operators 
        friend Matrix<T> operator+(const Matrix<T>& A, const Matrix<T>& B);
        friend Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B);
        friend Matrix<T> operator/(const Matrix<T>& A, const Matrix<T>& B);
        friend Matrix<T> operator-(const Matrix<T>& A, const Matrix<T>& B);

        // dot product [TODO: Add an exterior product]
        friend Matrix<T> operator%(const Matrix<T>& A, const Matrix<T>& B);
}