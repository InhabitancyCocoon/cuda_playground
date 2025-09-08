#include <iostream>
#include <cuda_bf16.h>

// nvcc -o bf16_demo bf16_demo.cu 

int main(int argc, char* argv[]) {
    float fp32_x = 2.455;
    nv_bfloat16 bf16_x = fp32_x;
    std::cout << float(bf16_x) << "\n";  // 2.45312
    return 0;
}