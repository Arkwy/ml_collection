#include <cstddef>
#include <hip/hip_runtime.h>
#include <iostream>

#include "tensor/device.hpp"
#include "tensor/tensor.hpp"

using namespace std;


int main() {
    Tensor<int, CPU> t1 = Tensor<int, CPU>::arange(12);
    t1.reshape({3, 2, 2});
    cout << t1.repr() << endl;
    for (std::size_t i = 0; i < t1.dim(); i++) {
      cout << t1.get_shape()[i] << ", ";
    }
    cout << endl;
    for (std::size_t i = 0; i < t1.dim(); i++) {
      cout << t1.get_stride()[i] << ", ";
    }
    cout << endl;

    return 0;
}
