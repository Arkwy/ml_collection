#ifndef PSO_H
#define PSO_H

#include <hip/amd_detail/amd_hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>

#include "../array/nd_array.hpp"
#include "../ops/reduce/1d.hpp"
#include "eval_function.hpp"
#include "particles.hpp"
#include "topology.hpp"


template <typename T>
concept EvalFunctionType = requires {
    // This checks if `T` is or inherits from any specialization of `EvalFunction<PEM, S, D>`
    []<PointEvaluationMode PEM, typename S, typename D>(EvalFunction<PEM, S, D>*) {}(std::declval<T*>());
};

template <size_t NumParticles, EvalFunctionType EF, TopologyType Tp>
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
        init_scores();
    }

  private:
    void init_scores() const {
        // TODO block/thread split and MultiThreaded/SingleThreaded/Custom support
        eval_function.eval_points<<<1, NumParticles>>>(
            particles.position.get_device(),
            particles.fitness.get_mut_device(),
            NumParticles
        );

        particles.best_position.device_copy(particles.position);
        particles.best_fitness.device_copy(particles.fitness);

        uint best_fitness_idx = reduce_1d<ArgMin<float>, NumParticles>(particles.fitness);

        float best_fitness = particles.fitness[best_fitness_idx];
        NDArray<float, EF::DefinitionSpace::dim> best_position = particles.position[best_fitness_idx];

        particles.best_known_fitness.fill(best_fitness);
        particles.best_known_position.fill(best_position);

        Tp::template share_fitness<NumParticles, EF::DefinitionSpace::dim>(
            particles.best_known_fitness,
            particles.best_known_position
        );
    }
};

#endif
