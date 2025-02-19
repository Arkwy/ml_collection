#include <iostream>
#include <hip/hip_runtime.h>

#include "tensor/device.hpp"
#include "tensor/tensor.hpp"

using namespace std;


int main() {

    Tensor<int, CPU> tc = Tensor<int, CPU>::full({10}, 5); 
    cout << tc.repr() << endl;
    const int* data = tc.get_storage().get_data();

    return 0;
}
