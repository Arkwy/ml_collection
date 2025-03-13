#ifndef PARTICLES_H
#define PARTICLES_H

#include <hip/amd_detail/amd_hip_runtime.h>

#include <cstddef>
#include <ostream>

#include "../array/nd_array.hpp"


/**
 * @brief ...
 *
 * ...
 * ...
 *
 * @param ...
 * @return ...
 */



template <size_t N, size_t D>
struct Particles {
    const NDArray<float, N, D> positions;
    const NDArray<float, N, D> velocities;

    Particles() : positions(), velocities() {}

    friend std::ostream& operator<<(std::ostream& os, const Particles<N, D>& particles) {
        os << N << " particles in a " << D << "D space:" << std::endl << std::endl;
        os << "Positions:" << std::endl;
        os << particles.positions << std::endl << std::endl;
        os << "Velocites:" << std::endl;
        os << particles.velocities << std::endl;
        return os;
    }
};

#endif
