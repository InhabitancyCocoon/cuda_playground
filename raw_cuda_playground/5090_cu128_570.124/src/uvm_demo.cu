/*

uvm stands for Unified Virtual Memory.


ref link:

https://nichijou.co/cudaRandom-UVA/

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-opt-in

https://docs.pytorch.org/docs/2.8/notes/cuda.html#using-custom-memory-allocators-for-cuda

Note that although explicitly data transfer between host and device is no longer needed,
under the hood there is still hardware-level page fault and data migration.

*/


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <iostream>


__global__ void write_value(int* ptr, int v) {
  *ptr = v;
}

int main() {
  int* ptr = nullptr;
  // Requires CUDA Managed Memory support
  cudaMallocManaged(&ptr, sizeof(int));
  write_value<<<1, 1>>>(ptr, 123);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", *ptr); 
  cudaFree(ptr); 
  return 0;
}