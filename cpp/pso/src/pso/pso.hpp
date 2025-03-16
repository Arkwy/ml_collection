#ifndef PSO_H
#define PSO_H

#include <hip/amd_detail/amd_hip_runtime.h>

#include <cstdint>

#include "../array/nd_array.hpp"
#include "eval_function.hpp"
#include "particles.hpp"


enum Topology {
    GLOBAL,
};

inline const char* topology_to_string(int topology) {
    switch (topology) {
        case GLOBAL:
            return "STAR";
            break;
        default:
            return "UNKNOWN";
    }
}

// Trait to check if a class is derived from a Space<D> whatever is D
template <typename T>
class is_eval_function {
    template <PointEvaluationMode PEM, SpaceType S>
    static std::true_type check(EvalFunction<PEM, S>*);  // Matches if T inherits Space<D> for any size_t D

    static std::false_type check(...);  // Maches any T if first overload doesn't match

  public:
    static constexpr bool value = decltype(check(std::declval<T*>()))::value;
};

template <typename T>
concept EvalFunctionType = is_eval_function<T>::value;

template <size_t NumParticles, EvalFunctionType EF, Topology Topology_>
struct PSO {
    const EF eval_function;
    const float momentum;
    const float cognitive_coefficient;
    const float social_coefficient;

    const Particles<NumParticles, EF::DefinitionSpace::dim> particles;

    PSO(const EF& eval_function,
        const float& weight,
        const float& cognitive_coefficient,
        const float& social_coefficient)
        : eval_function(eval_function),
          momentum(weight),
          cognitive_coefficient(cognitive_coefficient),
          social_coefficient(social_coefficient),
          particles() {
        eval_function.space.sample(particles.position);
        eval_function.space.sample(particles.velocity);
        init_connections();
        init_scores();
    }

  private:
    void init_connections() {
        if constexpr (Topology_ == Topology::GLOBAL) {
            particles.neighborhood.fill(0xffffffff);
        } else {
            static_assert(false, "Unexpected Topology!");
        }
    }


    static __global__ void share_scores(
        const float* const neighborhood, float* const best_known_fitness
    );  // TODO
    static __global__ void global_share_scores(
        float* const best_known_fitness,
        float* const best_known_position,
        const float* const fitness,
        const float* const position
    ) {}  // TODO

    void init_scores() const {
        // TODO block/thread split and MultiThreaded/SingleThreaded/Custom support
        eval_function.eval_points<<<1, NumParticles>>>(
            particles.position.get_device(),
            particles.fitness.get_mut_device(),
            NumParticles
        );

        particles.best_position.device_copy(particles.position);
        particles.best_fitness.device_copy(particles.fitness);
    }

    // self.fitness = self.space(self.position)

    // self.particle_best_fitness = self.fitness
    // self.particle_best_position = self.position

    // local_fitness = self.fitness[None, :].expand_as(self.neighborhood).where(
    //     self.neighborhood,
    //     torch.inf,
    // )

    // particle_best_known_fitness_id = local_fitness.argmin(-1)
    // self.particle_best_known_fitness = self.fitness[particle_best_known_fitness_id]
    // self.particle_best_known_position = self.position[particle_best_known_fitness_id]
};


#endif
