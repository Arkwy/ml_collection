#include <iostream>
#include <hip/hip_runtime.h>

#include "tensor/device.hpp"
#include "tensor/tensor.hpp"

using namespace std;


int main() {
  Tensor<int, CPU> t1 = Tensor<int, CPU>::arange(10).reshape({2, 5});
    cout << t1.repr() << endl;

    return 0;
}
