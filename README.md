## This is a CUDA related code playground.

### file structure

ubuntu 22.04, I put the command line command in the top comment of each file.

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
        "/usr/local/cuda/lib64"
    ],
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.cStandard": "c17"
}
```

- check [understanding ptx blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) for more tails

- setup nvcc

```
ls /usr/local/cuda/bin/

echo $PATH
export PATH=$PATH:/usr/local/cuda/bin

echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```