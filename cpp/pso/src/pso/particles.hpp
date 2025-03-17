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



/**
 * Particles for PSO algoritm.
 *
 * @member position
 * @member velocity
 * @member fitness  (evaluated by PSO)
 * @member neighborhood  (bit mask indicating for each directional pair of particles wether they share their position
 * and fitness)
 * @member best_fitness  (best fitness each particle ever reached)
 * @member best_position  (position corresponding to best fitness)
 * @member best_known_fitness  (best fitness a neighbor of each particle ever reached)
 * @member best_known_position  (position corresponding to best known fitness)
 */
template <uint N, uint D>
struct Particles {
    const NDArray<float, N, D> position;
    const NDArray<float, N, D> velocity;
    const NDArray<float, N> fitness;
    const NDArray<uint, (N + 31) / 32, (N + 31) / 32> neighborhood;
    const NDArray<float, N> best_fitness;
    const NDArray<float, N, D> best_position;
    const NDArray<float, N> best_known_fitness;
    const NDArray<float, N, D> best_known_position;

    Particles()
        : position(),
          velocity(),
          fitness(),
          neighborhood(),
          best_fitness(),
          best_position(),
          best_known_fitness(),
          best_known_position() {}

    friend std::ostream& operator<<(std::ostream& os, const Particles<N, D>& particles) {
        os << N << " particles in a " << D << "D space:" << std::endl << std::endl;
        os << "Positions:" << std::endl;
        os << particles.position << std::endl << std::endl;
        os << "Velocites:" << std::endl;
        os << particles.velocity << std::endl;
        return os;
    }
};

#endif
