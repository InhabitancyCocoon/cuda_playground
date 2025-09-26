## set up

- [nvidia cutlass](https://github.com/NVIDIA/cutlass)
- since cutlass / cute is a header-only library, you need to tell the compiler where the header file is.


```

git clone https://github.com/NVIDIA/cutlass.git

CC = nvcc
VERBOSE = --ptxas-options=-v
SUPPRESS_UNUSED_WARNING = -diag-suppress 177
CUTLASS_PATH = /root/autodl-tmp/cutlass
INCLUDE_CUTLASS = -I$(CUTLASS_PATH)/include
INCLUDE_CUTLASS_UTIL = -I$(CUTLASS_PATH)/tools/util/include

```
