## This is a CUDA related code playground.

### file structure

ubuntu 22.04, I put the command line command in the top comment of each file. In other words,
CMake is rarely used...  
When performance is not a concern, I will just use `nvcc -o main main.cu`.

```
-- cute
    -- ${cutlass version}
        -- ${gpu}_${cuda version}_${driver version}

-- cutlass
    -- ${cutlass version}
        -- ${gpu}_${cuda version}_${driver version}

-- raw cuda
    -- ${gpu}_${cuda version}_${driver version}
```

### setup note

- check [autodl github](https://www.autodl.com/docs/network_turbo/) for github access `source /etc/network_turbo`

- vscode c/cpp extension workspace settings example

```
{
    "C_Cpp.default.compilerPath": "/usr/bin/g++",
    "C_Cpp.default.systemIncludePath": [
        "/usr/local/cuda/include",
        "/usr/local/cuda/lib64",
        "/usr/include/c++/11",
        "/usr/include/x86_64-linux-gnu/c++/11",
        "/usr/include/c++/11/backward",
        "/usr/lib/gcc/x86_64-linux-gnu/11/include",
        "/usr/local/include",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include"
    ],
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.cStandard": "c17"
}
```


- setup nvcc

```
ls /usr/local/cuda/bin/

echo $PATH
export PATH=$PATH:/usr/local/cuda/bin

echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

- you can install the Nsight Visual Studio Code Edition extension.

- debug macro

```
#define CUDA_CHECK(call) \
{                         \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s: %d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}
```

- header files that maybe useful

```
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
```

### info

- check [understanding ptx blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) for more details

- check [nvidia tensor core evolution](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/) for more details

- cloud gpu (at least for autodl), is not happy with cuda kernel profiling tools such
as ncu.