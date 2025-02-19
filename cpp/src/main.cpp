#include <iostream>
#include <hip/hip_runtime.h>

#include "tensor/device.hpp"
#include "tensor/tensor.hpp"

using namespace std;


int main() {

    Tensor<bool, CPU> tc = Tensor<bool, CPU>::full({10}, 5); 
    cout << tc.repr() << endl;
    tc.reshape({5, 2});
    cout << tc.repr() << endl;
    tc.flatten();
    cout << tc.repr() << endl;
    tc.reshape({2, 5});
    cout << tc.repr() << endl;

    return 0;
}
