# CUDA-XLA (Accelerated Linear Algebra)
This is my own implementation of several linear algebra algorithms on CUDA. Currently the code only supports basic vector operations. In time, it will come to support SVD, Kernel tricks, matrix operations, and other operations. The code on this repository might not be the most efficient. I largely wrote it to gain familiarity with CUDA programming, and also to learn how linear algebra libraries worked under the hood.  

You can build the project by running:
```
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES="YOUR GPU ARCHITECTURE TYPE" CUDA-XLA
make
./xla
```
I ran all of this on Google Collab using a Python notebook and using cell-magics. There the free GPU is an NVIDIA T4, so the command would be ```cmake -DCMAKE_CUDA_ARCHITECTURES="75" CUDA-XLA```
