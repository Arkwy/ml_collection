#ifndef PSO_HIP
#define PSO_HIP

#include <hip/amd_detail/amd_hip_cooperative_groups.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>
#include <rocrand/rocrand_uniform.h>

#include <limits>

#include "eval_function.hpp"

using namespace cooperative_groups;

// State of PSO
struct __align__(16) State {
    uint iterations;
    bool initialized;
    uint seed;
    const uint dim;
    const uint n;
    const float momentum;
    const float cognitive_coefficient;
    const float social_coefficient;
    float* const position;
    float* const velocity;
    float* const fitness;
    float* const best_fitness;
    float* const best_position;
    float* const best_known_fitness;
    float* const best_known_position;
    float* const r_p;
    float* const r_g;
};



// Initilize particles position and velocities by sampling random values in [-1; 1]
// Initilize best_fitness and best_known_fitness to inf so they will be replaced at 1st iteration
__device__ void init(const State& state, const uint& t_idx, rocrand_state_xorwow& rng_state) {
    for (uint i = 0; i < state.dim; i++) {
        state.position[t_idx * state.dim + i] = rocrand_uniform(&rng_state) * 2. - 1.;
        state.velocity[t_idx * state.dim + i] = rocrand_uniform(&rng_state) * 2. - 1.;
    }
    state.best_fitness[t_idx] = std::numeric_limits<float>::infinity();
    state.best_known_fitness[t_idx] = std::numeric_limits<float>::infinity();
}


// Compute the fitness according to particles postions and user defined evaluation function
template <EvalFunction EF>
__device__ void compute_fitness(const State& state, const uint& t_idx) {
    state.fitness[t_idx] = EF::eval(state.position + t_idx * state.dim);
}


// Update particle best historical position and fitness
__device__ void update_best(const State& state, const uint& t_idx) {
    if (state.fitness[t_idx] < state.best_fitness[t_idx]) {
        state.best_fitness[t_idx] = state.fitness[t_idx];
        for (uint i = 0; i < state.dim; i++) {
            state.best_position[t_idx * state.dim + i] = state.position[t_idx * state.dim + i];
        }
    }
}


// Update neighborhood best historical position and fitness (star topology)
// TODO add new topologies
__device__ void share(const State& state, const uint& t_idx) {
    if (state.fitness[t_idx] < state.best_known_fitness[t_idx]) {
        if (state.fitness[0] < state.fitness[t_idx]) {
            state.best_known_fitness[t_idx] = state.fitness[0];
            for (uint i = 0; i < state.dim; i++) {
                state.best_known_position[t_idx * state.dim + i] = state.position[i];
            }
        } else {
            state.best_known_fitness[t_idx] = state.fitness[t_idx];
            for (uint i = 0; i < state.dim; i++) {
                state.best_known_position[t_idx * state.dim + i] = state.position[t_idx * state.dim + i];
            }
        }
    }
}


// Update current position and velocity
__device__ void update_particle_state(const State& state, const uint& t_idx, rocrand_state_xorwow& rng_state) {
    state.r_p[t_idx] = rocrand_uniform(&rng_state);
    state.r_g[t_idx] = rocrand_uniform(&rng_state);
    for (uint i = 0; i < state.dim; i++) {
        state.velocity[t_idx * state.dim + i] =
            (state.momentum * state.velocity[t_idx * state.dim + i]) +

            (state.cognitive_coefficient * state.r_p[t_idx] *
             (state.best_position[t_idx * state.dim + i] - state.position[t_idx * state.dim + i])) +

            (state.social_coefficient * state.r_g[t_idx] *
             (state.best_known_position[t_idx * state.dim + i] - state.position[t_idx * state.dim + i]));

        state.position[t_idx * state.dim + i] += state.velocity[t_idx * state.dim + i];
    }
}


// Main PSO kernel, initilize particles and run PSO over specified iterations.
// The kernel do the work of index checking, synchronization and shared_memory data transfers (TODO) for the device
// functions (maybe not for `share` as requirements are topology dependant ?? TODO).
template <EvalFunction EF>
__global__ void pso_kernel(const State* state_ptr) {
    State state = *state_ptr;
    thread_group gg = this_grid();

    uint t_idx = gg.thread_rank();

    // TODO rm this check
    uint _t_idx = threadIdx.x + blockDim.x * blockIdx.x;
    assert(t_idx == _t_idx);

    rocrand_state_xorwow rng_state;
    rocrand_init(state.seed, t_idx, 0, &rng_state);

    const bool participates = t_idx < state.n;  // whether thread participates in some computations

    if (!state.initialized) {
        if (participates) init(state, t_idx, rng_state);
        if (participates) compute_fitness<EF>(state, t_idx);
        if (participates) update_best(state, t_idx);
        gg.sync();
        if (participates) share(state, t_idx);
        gg.sync();
    }

    for (uint i = 0; i < state.iterations; i++) {
        if (participates) update_particle_state(state, t_idx, rng_state);
        if (participates) compute_fitness<EF>(state, t_idx);
        if (participates) update_best(state, t_idx);
        gg.sync();
        if (participates) share(state, t_idx);
        gg.sync();
    }
}


#endif
