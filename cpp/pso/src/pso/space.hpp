#ifndef SPACE_H
#define SPACE_H

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_warp_functions.h>

#include <cstddef>
#include <cstdint>
#include <rocrand/rocrand.hpp>
#include <utility>

#include "../array/nd_array.hpp"
#include "../utils/logger.hpp"


__device__ inline float clamp(float x, float min_val, float max_val) { return fmaxf(min_val, fminf(x, max_val)); }

template <uint N, uint D>
__global__ void scale_in_bounds(float* const points, const float* const bounds) {
    uint x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint y_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_idx < D && y_idx < N) {
        uint idx = y_idx * D + x_idx;
        points[idx] *= bounds[2 * x_idx + 1] - bounds[2 * x_idx];
        points[idx] += bounds[2 * x_idx];
    }
}


template <uint N, uint D>
__global__ void clamp(float* const points, const float* const bounds) {
    int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int y_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_idx < D && y_idx < N) {
        int idx = y_idx * D + x_idx;
        points[idx] = clamp(points[idx], bounds[2 * x_idx], bounds[2 * x_idx + 1]);
    }
}

template <uint D>
struct Space {
    static constexpr uint dim = D;

    template <uint N>
    void sample(const NDArray<float, N, D>& points) const;

    template <uint N>
    void bound(const NDArray<float, N, D>& points) const;


  protected:
    Space() = default;
    std::pair<dim3, dim3> kernel_dims_pointwise_ops(const uint& points, const uint& device_id) const {
        dim3 grid_dim;
        dim3 block_dim;

        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device_id));

        uint warp_size = props.warpSize;
        uint max_threads_per_block = props.maxThreadsPerBlock;

        uint x_warps = (D + warp_size - 1) / warp_size;
        uint x_threads = x_warps * warp_size;

        // row major arrays, grouping along last = D dim for better coalescence
        // 1st case: can have mutiple points in one block as D is small enough to fit multiple times in a block
        // 2nd case: each block process at most a single point + partially the following/previous one
        if (x_threads <= (max_threads_per_block / 2)) {
            uint y_threads_per_block;
            if (x_warps == 1) {
                while (x_threads >> 1 >= D) {
                    x_threads >>= 1;
                }
            }

            y_threads_per_block = std::min((uint)points, max_threads_per_block / x_threads);

            block_dim.x = x_threads;
            block_dim.y = y_threads_per_block;
            grid_dim.y = (points + y_threads_per_block - 1) / y_threads_per_block;

        } else {
            block_dim.x = std::min(x_threads, max_threads_per_block);
            grid_dim.x = (x_threads + max_threads_per_block - 1) / max_threads_per_block;
            grid_dim.y = points;
        }


        return std::pair<dim3, dim3>(grid_dim, block_dim);
    }
};


template <uint D>
struct BoxSpace : public Space<D> {
    const NDArray<float, D, 2> bounds;

    BoxSpace(const std::array<std::array<float, 2>, D>& bounds) : bounds() {
        for (uint i = 0; i < D; i++) {
            this->bounds[i][0] = bounds[i][0];
            this->bounds[i][1] = bounds[i][1];
        }
    }

    template <uint N>
    void sample(const NDArray<float, N, D>& points) const {
        HIP_CHECK(hipSetDevice(points.device_id()));

        rocrand_cpp::default_random_engine engine;
        rocrand_cpp::uniform_real_distribution distribution;

        distribution(engine, points.get_mut_device(), N * D * sizeof(float));

        auto [grid_dim, block_dim] = this->kernel_dims_pointwise_ops(N, points.device_id());

        LOG(LOG_LEVEL_INFO,
            "\nSampling %d points in a %dD space:\ngrid dim: (%d, %d, %d)\nblock dim: (%d, %d, %d)\n",
            N,
            D,
            grid_dim.x,
            grid_dim.y,
            grid_dim.z,
            block_dim.x,
            block_dim.y,
            block_dim.z);

        scale_in_bounds<N, D><<<grid_dim, block_dim>>>(points.get_mut_device(), bounds.get_device());
    }

    template <uint N>
    void bound(const NDArray<float, N, D>& points) const {
        HIP_CHECK(hipSetDevice(points.device_array->device_id()))

        auto [grid_dim, block_dim] = this->kernel_dims_pointwise_ops(N);

        clamp<N, D><<<grid_dim, block_dim>>>(points.get_mut_device(), bounds.get_device());
    }
};

#endif
