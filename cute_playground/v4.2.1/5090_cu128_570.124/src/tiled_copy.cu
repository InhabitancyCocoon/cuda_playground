/*

Ref link:

https://github.com/NVIDIA/cutlass/blob/v4.2.1/examples/cute/tutorial/tiled_copy.cu

*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"



template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout) {
    using namespace cute;

}


template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, Tiled_Copy tiled_copy) {
    using namespace cute;

    // Slice the tensors to obtain a view into each tile.
    Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);
    Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);

    // Constructing a tensor according to each thread's slice.
    ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

    Tensor thr_tile_S = thr_copy.partition_S(tile_S);    // (CopyOp, CopyM, CopyN)
    Tensor thr_tile_D = thr_copy.partition_S(tile_D);    // (CopyOp, CopyM, CopyN)

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_fragment because the first mode is the instruction-local mode

    Tensor fragment = make_fragment_like(thr_tile_D);    // (CopyOp, CopyM, CopyN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(tiled_copy, thr_tile_S, fragment);
    copy(tiled_copy, fragment, thr_tile_D);
}


int main(int argc, char* argv[]) {
    using namespace cute;
    using Element = float;
    

    auto tensor_shape = make_shape(256, 512);

    thrust::host_vector<Element> h_S(size(tensor_shape));
    thrust::host_vector<Element> h_D(size(tensor_shape));

    for (size_t i = 0; i < h_S.size(); ++i) {
        h_S[i] = static_cast<Element>(i);
        h_D[i] = Element{};
    }

    thrust::device_vector<Element> d_S = h_S;
    thrust::device_vector<Element> d_D = h_D;

    Tensor tensor_S = make_tensor(
        make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())),
        make_layout(tensor_shape)
    );

    Tensor tensor_D = make_tensor(
        make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())),
        make_layout(tensor_shape)
    );

    auto block_shape = make_shape(Int<128>{}, Int<64>{});


    if (size<0>(tensor_shape) % size<0>(block_shape) || size<1>(tensor_shape) % size<1>(block_shape)) {
        std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
        return -1;
    }

    if (not evenly_divides(tensor_shape, block_shape)) {
        std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
        return -1;
    }

    // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
    // shape, and modes (m', n') correspond to the number of tiles.
    //
    // These will be used to determine the CUDA kernel grid dimensions.
    // I guess: (256, 512) ==> ((128, 64), 2, 8)
    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);

    // Construct a TiledCopy with a specific access pattern.
    //   This version uses a
    //   (1) Layout-of-Threads to describe the number and arrangement of threads (e.g. row-major, col-major, etc),
    //   (2) Layout-of-Values that each thread will access.    


    // Thread arrangement
    Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));


    // Value arrangement per thread
    Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));


    // Define `AccessType` which controls the size of the actual memory access instruction.
    using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;  // A very specific access width copy instruction
    // using CopyOp = UniversalCopy<cutlass::AlignedArray<Element, size(val_layout)>>;  // A more generic type that supports many copy strategies
    // using CopyOp = AutoVectorizingCopy;  // An adaptable-width instruction that assumes maximal alignment of inputs

    // A Copy_Atom corresponds to one CopyOperation applied to Tensors of type Element.
    using Atom = Copy_Atom<CopyOp, Element>;

    // Construct tiled copy, a tiling of copy atoms.
    //
    // Note, this assumes the vector and thread layouts are aligned with contiguous data
    // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
    // reads. Alternative value layouts are also possible, though incompatible layouts
    // will result in compile time errors.

    TiledCopy tiled_copy = make_tiled_copy(
        Atom{},          // Access strategy
        thr_layout,      // (32, 8)
        val_layout       // (4, 1)
    );


    // Determine grid and block dimensions.
    dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
    dim3 blockDim(size(thr_layout));

    // Launch the kernel.

    copy_kernel_vectorized<<<gridDim, blockDim>>>(
        tiled_tensor_S,
        tiled_tensor_D,
        tiled_copy
    );

    cudaError result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
        return -1;
    }


    // verify
    h_D = d_D;

    int32_t errors = 0;
    int32_t const kErrorLimit = 10;

    for (size_t i = 0; i < h_D.size(); ++i) {
        if (h_S[i] != h_D[i]) {
            std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;
            if (++errors >= kErrorLimit) {
                std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
                return -1;
            }
        }
    }

    std::cout << "Success.\n";

    return 0;


}