#ifndef OPS_REDUCE_1D_H
#define OPS_REDUCE_1D_H

#include <hip/hip_runtime.h>

#include <bit>

#include "../../array/device_array.hpp"
#include "../../array/nd_array.hpp"
#include "../../../utils/hip_utils.hpp"
#include "../fill.hpp"
#include "ops.hpp"


template <typename Op>
    requires is_no_id_op<Op>
__device__ Op::DType reduce_block_1d(const typename Op::DType* const data, uint size) {
    uint gt_idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint lt_idx = threadIdx.x;

    extern __shared__ typename Op::DType s_data[];  // TODO consider static allocation

    // TODO merge with 1st loop iter and divide number of threads by two
    s_data[lt_idx] = (gt_idx < size) ? data[gt_idx] : Op::identity;
    __syncthreads();

    for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lt_idx < stride) {
            s_data[lt_idx] = Op::run(s_data[lt_idx], s_data[lt_idx + stride]);
        }
        __syncthreads();
    }

    return (lt_idx == 0) ? s_data[0] : Op::identity;
}


template <typename Op>
    requires is_no_id_op<Op>
__global__ void kernel_reduce_block_1d(typename Op::DType* const data, uint size) {
    typename Op::DType val = reduce_block_1d<Op>(data, size);
    __syncthreads();
    if (threadIdx.x == 0) {
        data[blockIdx.x] = val;
    }
}


template <typename Op>
    requires is_id_op<Op>
__global__ void kernel_reduce_block_1d(typename Op::DType* const val, uint* const idx, uint size) {
    uint gt_idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint lt_idx = threadIdx.x;

    extern __shared__ typename Op::DType s_data[];  // TODO consider static allocation

    typename Op::DType* s_val = s_data;
    uint* s_idx = (uint*)(s_data + blockDim.x);

    // TODO merge with 1st loop iter and divide number of threads by two
    s_val[lt_idx] = (gt_idx < size) ? val[gt_idx] : Op::identity;
    s_idx[lt_idx] = (gt_idx < size) ? idx[gt_idx] : 0;
    __syncthreads();

    for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lt_idx < stride) {
            Op::run(s_val[lt_idx], s_val[lt_idx + stride], s_idx[lt_idx], s_idx[lt_idx + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        val[blockIdx.x] = s_val[0];
        idx[blockIdx.x] = s_idx[0];
    }
}

// TODO reduce boiler plate code
template <typename Op, uint N>
    requires is_no_id_op<Op>
Op::DType reduce_1d(const NDArray<typename Op::DType, N>& data) {
    const uint device_id = data.device_id();
    HIP_CHECK(hipSetDevice(device_id));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    const uint max_threads_per_block = props.maxThreadsPerBlock;
    if (!std::has_single_bit(max_threads_per_block)) {
        throw std::runtime_error("Max threads per block is not a power of 2, case is not supported.");
    }

    uint blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    uint n = N;

    DeviceArray<typename Op::DType> data_arr(N);
    HIP_CHECK(
        hipMemcpy(data_arr.data, data.get_device(), data.size * sizeof(typename Op::DType), hipMemcpyDeviceToDevice)
    );  // TODO this copy could be avoided
    HIP_CHECK(hipDeviceSynchronize());


    uint threads_per_block;

    do {
        threads_per_block = std::min(max_threads_per_block, std::bit_ceil(n));
        hipLaunchKernelGGL(
            (kernel_reduce_block_1d<Op>),
            dim3(blocks),
            dim3(threads_per_block),
            threads_per_block * sizeof(typename Op::DType),
            0,
            data_arr.data,
            n
        );
        n = blocks;
        blocks = (blocks + max_threads_per_block - 1) / max_threads_per_block;
        HIP_CHECK(hipDeviceSynchronize());
    } while (n > 1);

    typename Op::DType result;
    HIP_CHECK(hipMemcpy(&result, data_arr.data, sizeof(typename Op::DType), hipMemcpyDeviceToHost));

    return result;
}



template <typename Op, uint N>
    requires is_id_op<Op>
uint reduce_1d(const NDArray<typename Op::DType, N>& data) {
    const uint device_id = data.device_id();
    HIP_CHECK(hipSetDevice(device_id));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    const uint max_threads_per_block = props.maxThreadsPerBlock;
    if (!std::has_single_bit(max_threads_per_block)) {
        throw std::runtime_error("Max threads per block is not a power of 2, case is not supported.");
    }

    uint blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    uint n = N;

    DeviceArray<typename Op::DType> data_arr(N);
    HIP_CHECK(
        hipMemcpy(data_arr.data, data.get_device(), data.size * sizeof(typename Op::DType), hipMemcpyDeviceToDevice)
    );  // TODO this copy could be avoided

    DeviceArray<uint> idx_arr(n);
    hipLaunchKernelGGL(arange, dim3(blocks), dim3(std::min(max_threads_per_block, n)), 0, 0, idx_arr.data, n);
    HIP_CHECK(hipDeviceSynchronize());

    uint threads_per_block;

    do {
        threads_per_block = std::min(max_threads_per_block, std::bit_ceil(n));
        hipLaunchKernelGGL(
            (kernel_reduce_block_1d<Op>),
            dim3(blocks),
            dim3(threads_per_block),
            threads_per_block * (sizeof(typename Op::DType) + sizeof(uint)),
            0,
            data_arr.data,
            idx_arr.data,
            n
        );
        n = blocks;
        blocks = (blocks + max_threads_per_block - 1) / max_threads_per_block;
        HIP_CHECK(hipDeviceSynchronize());
    } while (n > 1);

    uint result;
    HIP_CHECK(hipMemcpy(&result, idx_arr.data, sizeof(uint), hipMemcpyDeviceToHost));

    return result;
}

#endif
