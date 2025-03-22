#include "fill.hpp"

#include <hip/hip_runtime.h>

__global__ void arange(uint* const data, uint size) {
    uint t_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (t_idx < size) {
        data[t_idx] = t_idx;
    }
}
