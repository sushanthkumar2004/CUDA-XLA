#pragma once
#include <vector>

template <typename T>
class Vec {
    private: 
        size_t size; 
        // points to a location on device NOT host
        T* data_ptr; 

    public:
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, 
                    "Vec<T> can only be instantiated with float or double.");

        // initialize vector of zeros 
        Vec(size_t size);

        // initialize from vector of values 
        Vec(const std::vector<T>& vec);

        // destructor for device memory 
        ~Vec(); 

        // element-wise operators 
        // friend Vec<T> operator+(const Vec<T>& A, const Vec<T>& B);
        // friend Vec<T> operator*(const Vec<T>& A, const Vec<T>& B);
        // friend Vec<T> operator/(const Vec<T>& A, const Vec<T>& B);
        // friend Vec<T> operator-(const Vec<T>& A, const Vec<T>& B);

        // dot product [TODO: Add an exterior product]
        template <typename U> 
        friend U operator%(const Vec<U>& A, const Vec<U>& B);
};