#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>


template <typename T>
struct HIPTensor {
	T* const data;
	const size_t dim;
	const size_t numel;
	const size_t* const shape;
	const size_t* const stride;
	const size_t offset;
};


template <typename T>
__global__ void write_to_contiguous(HIPTensor<T> src, HIPTensor<T> dest) {
	// assumes dim, numel and shape of src and dest are the sames

    assert(blockDim.y == 1 && blockDim.z == 1);
    assert(blockIdx.y == 0 && blockIdx.z == 0);
    assert(threadIdx.y == 0 && threadIdx.z == 0);

	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int dest_idx = thread_idx;

	int src_idx = src.offset;
	for (size_t i = 0; i < src.dim; i++) {
		size_t q = thread_idx / dest.stride[i];
		src_idx += q * src.stride[i];
		thread_idx -= q * dest.stride[i];
	}
    
	if (dest_idx < src.numel) {
		dest.data[dest_idx] = src.data[src_idx];
	}
}

template <typename T>
__global__ void write_from_contiguous(HIPTensor<T> src, HIPTensor<T> dest) {
	// assumes dim, numel and shape of src and dest are the sames

    assert(blockDim.y == 1 && blockDim.z == 1);
    assert(blockIdx.y == 0 && blockIdx.z == 0);
    assert(threadIdx.y == 0 && threadIdx.z == 0);

	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int src_idx = thread_idx;

	int dest_idx = dest.offset;
	for (size_t i = 0; i < src.dim; i++) {
		size_t q = thread_idx / src.stride[i];
		dest_idx += q * dest.stride[i];
		thread_idx -= q * src.stride[i];
	}
    
	if (src_idx < src.numel) {
		dest.data[dest_idx] = src.data[src_idx];
	}
}

template <typename T>
__global__ void fill_discontiguous(HIPTensor<T> tensor, T value) {
	// assumes provided stride is wihout permutations

    assert(blockDim.y == 1 && blockDim.z == 1);
    assert(blockIdx.y == 0 && blockIdx.z == 0);
    assert(threadIdx.y == 0 && threadIdx.z == 0);

	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx; // = TODO

	if (thread_idx < tensor.numel) {
        tensor.data[idx] = value;
	}

}
