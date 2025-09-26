CC = nvcc
VERBOSE = --ptxas-options=-v
SUPPRESS_UNUSED_WARNING = -diag-suppress 177
CUTLASS_PATH = /root/autodl-tmp/cutlass
INCLUDE_CUTLASS = -I$(CUTLASS_PATH)/include
INCLUDE_CUTLASS_UTIL = -I$(CUTLASS_PATH)/tools/util/include


ARCH = -arch=compute_120
.PHONY: clean


build/tiled_copy.out: src/tiled_copy.cu
	$(CC) $(ARCH) $(INCLUDE_CUTLASS) $(INCLUDE_CUTLASS_UTIL) src/tiled_copy.cu -o build/tiled_copy.out


clean:
	@echo "Removing object files..."
	rm -f build/*.out
	rm -f build/*.o
