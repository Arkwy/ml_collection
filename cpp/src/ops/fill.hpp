#ifndef OPS_FILL_H
#define OPS_FILL_H

#include <hip/hip_runtime.h>

__global__ void arange(uint* const data, uint size);

#endif
