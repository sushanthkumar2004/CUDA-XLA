#include <vector>

template <typename T>
class Vec {
    private: 
        size_t size; 
        // points to a location on device NOT host
        T* data_ptr; 

    public:
        // initialize vector of zeros 
        Vec(size_t size);

        // initialize from vector of values 
        Vec(std::vector<T> vec);

        // destructor for device memory 
        ~Vec(); 

        T norm() {
            return (T) 0; 
        }

        // element-wise operators 
        // friend Vec<T> operator+(const Vec<T>& A, const Vec<T>& B);
        // friend Vec<T> operator*(const Vec<T>& A, const Vec<T>& B);
        // friend Vec<T> operator/(const Vec<T>& A, const Vec<T>& B);
        // friend Vec<T> operator-(const Vec<T>& A, const Vec<T>& B);

        // dot product [TODO: Add an exterior product]
        friend T operator%(const Vec<T>& A, const Vec<T>& B); 
}

template <typename T>
T operator%(const Vec<T>& A, const Vec<T>& B);