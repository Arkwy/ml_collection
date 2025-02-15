#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdio>

using namespace std;

class A{
private:
    size_t n;
    int* arr;
public:
    A(size_t n) n(n) {
        arr = new int[n];
    }

    int* getArray() {
        return arr;  // Allows read and write access, but cannot delete the array
    }

    ~A() {
        delete[] arr;
    }
};


int main() {

    return 0;
}
