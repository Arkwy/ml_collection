#ifndef PSO_TOPOLOGY_H
#define PSO_TOPOLOGY_H

#include <hip/hip_runtime.h>

#include "../array/nd_array.hpp"
#include "../ops/reduce/1d.hpp"

enum class TopologyCategory {
    GLOBAL,
};

template <TopologyCategory T>
struct Topology {
    constexpr static const TopologyCategory category = T;

    template <size_t N, size_t D>
    static void share_fitness(
        const NDArray<float, N, D>& best_known_fitness, const NDArray<float, N, D>& best_known_position
    );
};

template <>
struct Topology<TopologyCategory::GLOBAL> {
    template <size_t N, size_t D>
    static void share_fitness(
        const NDArray<float, N>& best_known_fitness, const NDArray<float, N, D>& best_known_position
    ) {
        uint best_fitness_idx = reduce_1d<ArgMin<float>, N>(best_known_fitness);
        best_known_fitness.fill(best_known_fitness[best_fitness_idx]);
        best_known_position.fill(best_known_position[best_fitness_idx]);
    }
};


inline const char* topology_to_string(TopologyCategory topology_type) {
    switch (topology_type) {
        case TopologyCategory::GLOBAL:
            return "GLOBAl";
            break;
        default:
            return "UNKNOWN";
    }
}


template <typename T>
concept TopologyType = requires {
    // This checks if `T` is or inherits from any specialization of `Topology<Tp>`
    []<TopologyCategory Tp>(Topology<Tp>*) {}(std::declval<T*>());
};


#endif
