#include <iostream>
#include <hip/hip_runtime.h>

#include "tensor/device.hpp"
#include "tensor/tensor.hpp"

using namespace std;


int main() {
  Tensor<int, CPU, 1> t1 = Tensor<int, CPU, 1>::arange(10);
    cout << t1.repr() << endl;
    // cout << t1.reshape({10}).repr() << endl;

    return 0;
}
