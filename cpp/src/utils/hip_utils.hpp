#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include <iostream>

#define HIP_CHECK(command)                                                                                             \
    {                                                                                                                  \
        hipError_t status = command;                                                                                   \
        if (status != hipSuccess) {                                                                                    \
            std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl;                              \
            std::abort();                                                                                              \
        }                                                                                                              \
    }

#endif
