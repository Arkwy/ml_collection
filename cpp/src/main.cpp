#include <iostream>
#include <hip/hip_runtime.h>

#include "tensor/device.hpp"
#include "tensor/tensor.hpp"

using namespace std;


int main() {

    Tensor<float, CPU> tc = Tensor<float, CPU>::full({4, 5, 3}, 0.1343); 
    cout << tc.repr() << endl;

    return 0;
}
