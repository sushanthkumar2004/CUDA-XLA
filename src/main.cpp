#include <iostream>
#include "gpu.hpp"
#include "vec.hpp"
#include <vector>

int main()
{
    std::cout << "Hello, world!" << std::endl;

#ifdef USE_CUDA
    std::cout << "CUDA: On" << std::endl;
    printCudaVersion();
#else
    std::cout << "CUDA: Off" << std::endl;
#endif

    std::vector<double> vec; 
    for (int i=0; i<10000; i++) {
        vec.push_back(1.0); 
    }

    Vec<double> vector((size_t) 200); 

    // double product = vector%vector;
    // std::cout << product << std::endl; 

    return 0;
}
