#include <iostream>
#include <hip/hip_runtime.h>

// #include "utils/hip_timer.hpp"
// #include "utils/hip_utils.hpp"
#include "tensor/device.hpp"
#include "tensor/tensor.hpp"

using namespace std;

// __global__ void kernel(const float* a, const float* b, float *c, int n) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     // Simulate work
//     if (idx < n) {
//         c[idx] = a[idx] + b[idx];
//     }
// }

int main() {
    // HIPTimer timer = HIPTimer();

    // int n = 1000;
    // size_t N = sizeof(float) * n;
    // float *h_a = (float*) malloc(N);
    // float *h_b = (float*) malloc(N);
    // float *h_c = (float*) malloc(N);

    // for (int i = 0; i < n; i++) {
    //     h_a[i] = i;
    //     h_b[i] = 2*i;
    // }

    // float *d_a, *d_b, *d_c;
    // HIP_CHECK(hipMalloc(&d_a, N));
    // HIP_CHECK(hipMalloc(&d_b, N));
    // HIP_CHECK(hipMalloc(&d_c, N));
    
    // HIP_CHECK(hipMemcpy(d_a, h_a, N, hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(d_b, h_b, N, hipMemcpyHostToDevice));

    // timer.start();
    // kernel<<<1, 1024>>>(d_a, d_b, d_c, n);
    // timer.stop();

    // HIP_CHECK(hipMemcpy(h_c, d_c, N, hipMemcpyDeviceToHost));
     

    // for (int i = 0; i < n-1; i++) {
    //     std::cout << h_c[i] << ", ";
    // }
    // std::cout << h_c[n-1] << std::endl;

    // timer.status();

    // HIP_CHECK(hipFree(d_a));
    // HIP_CHECK(hipFree(d_b));
    // HIP_CHECK(hipFree(d_c));

    // free(h_a);
    // free(h_b);
    // free(h_c);

    Tensor<float, CPU> tf = Tensor<float, CPU>({4, 5, 3});
    // Tensor<float> te = tf;

    cout << tf.repr() << endl;
    // cout << tc.repr() << endl;

    return 0;
}
