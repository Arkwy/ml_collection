#include <hip/hip_runtime.h>


template <typename T>
struct Tensor {
	T* const data;
	const size_t dim;
	const size_t numel;
    const size_t* const shape;
	const size_t* const stride;
    const size_t offset;
};



template <typename T>
__global__ void write_to_contiguous(const Tensor<T>& src, const Tensor<T>& dest) {

    // assumes dim, numel and shape of src and dest are the sames

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // dest is contiguous so storage index is straight forward
	int dest_idx = x * y * z;

    // not src ...
	int src_idx = src.offset + x * src.stride[0] + y * src.stride[1];
	if (src.dim == 3) {
		src_idx += z * src.stride[1];
		dest_idx += z * dest.stride[1];
	} else if (src.dim > 3) { // especially if dim > 3.
		for (size_t i = 3; i < src.dim; i++) {
            dest_idx += (z % dest.stride[i]) * src.stride[i];
		}
	}

	if (dest_idx < src.numel) {
        dest[dest_idx] = src[src_idx];
	}
}
