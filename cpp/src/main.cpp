#include <iostream>
#include <hip/hip_runtime.h>

#include "tensor/device.hpp"
#include "tensor/tensor.hpp"

using namespace std;


int main() {

    int* a;
    {
        vector<int> b({1, 2, 3, 4});
        a = b.data();
        cout << a[0] << endl;
    }
    cout << a[0] << endl;

    return 0;
}
