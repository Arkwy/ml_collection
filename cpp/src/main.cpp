#include <cstddef>
#include <hip/hip_runtime.h>
#include <iostream>

#include "tensor/device.hpp"
#include "tensor/tensor.hpp"

using namespace std;


int main() {
    Tensor<int, CPU> t1 = Tensor<int, CPU>::arange(12);

    t1.reshape({2, 2, 3});
    cout<< endl << t1.repr() << endl;

    t1.permute({2, 1, 0});
    cout<< endl << t1.repr() << endl;

    t1.reshape({2,2,3});
    cout<< endl << t1.repr() << endl;

    Tensor<int, CPU> t2 = t1.clone();
    cout<< endl << t2.repr() << endl;

    return 0;
}
