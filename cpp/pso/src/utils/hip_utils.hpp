#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include <stdexcept>

#include "logger.hpp"

#define HIP_CHECK(command)                                                                                            \
    {                                                                                                                 \
        hipError_t status = command;                                                                                  \
        if (status != hipSuccess) {                                                                                   \
            LOG(LOG_LEVEL_ERROR, "Error: HIP reports %s. %s, line %d.", hipGetErrorString(status), __FILE__, __LINE__ \
            );                                                                                                        \
            throw std::runtime_error("HIP error");                                                                    \
        }                                                                                                             \
    }

#endif
