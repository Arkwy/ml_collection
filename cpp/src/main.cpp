#include <cstddef>
#include <hip/hip_runtime.h>
#include <iostream>

#include "tensor/device.hpp"
#include "tensor/tensor.hpp"

using namespace std;


int main() {
    Tensor<int, GPU> t1 = Tensor<int, GPU>::arange(12000);


    t1.reshape({2, 2, 3000});
    // cout<< endl << t1.repr() << endl;

    t1.permute({2, 1, 0});
    // cout<< endl << t1.repr() << endl;

    t1.reshape({2,2,3000});
    // cout<< endl << t1.repr() << endl;
    Tensor<int, CPU> t1c = t1.to(CPU());
    // cout<< endl << t1c.repr() << endl;

    Tensor<int, GPU> t2 = t1.clone();
    // cout<< endl << t2.repr() << endl;
    Tensor<int, CPU> t2c = t2.to(CPU());
    // cout<< endl << t2c.repr() << endl;

    return 0;
}
