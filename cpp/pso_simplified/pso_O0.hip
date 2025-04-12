#ifndef PSO_HIP
#define PSO_HIP

#include <hip/amd_detail/amd_hip_cooperative_groups.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>
#include <rocrand/rocrand_uniform.h>

#include <bit>
#include <limits>

#include "eval_function.hpp"

#ifdef WARP_SIZE
static_assert(std::is_integral<decltype(WARP_SIZE)>::value, "`WARP_SIZE` must be an integral type.");
static_assert(WARP_SIZE >= 1, "`WARP_SIZE` must be >= 1.");

#ifdef MAX_THREADS_PER_BLOCK
static_assert(
    std::is_integral<decltype(MAX_THREADS_PER_BLOCK)>::value, "`MAX_THREADS_PER_BLOCK` must be an integral type."
);
static_assert(MAX_THREADS_PER_BLOCK >= 1, "`MAX_THREADS_PER_BLOCK` must be >= 1.");


using namespace cooperative_groups;


// State of PSO
struct __align__(16) State {
    // Kernel launch specific args
    uint iterations;
    bool initialized;
    uint seed;
    // Hyperparameters
    const uint dim;
    const uint n;
    const float momentum;
    const float cognitive_coefficient;
    const float social_coefficient;
    // PSO state
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
__device__ void init(const State& state, const uint& g_idx, rocrand_state_xorwow& rng_state) {
    for (uint i = 0; i < state.dim; i++) {
        state.position[g_idx * state.dim + i] = rocrand_uniform(&rng_state) * 2. - 1.;
        state.velocity[g_idx * state.dim + i] = rocrand_uniform(&rng_state) * 2. - 1.;
    }
    state.best_fitness[g_idx] = std::numeric_limits<float>::infinity();
    state.best_known_fitness[g_idx] = std::numeric_limits<float>::infinity();
}


// Compute the fitness according to particles postions and user defined evaluation function
template <EvalFunction EF>
__device__ void compute_fitness(const State& state, const uint& g_idx) {
    state.fitness[g_idx] = EF::eval(state.position + g_idx * state.dim);
}


// Update particle best historical position and fitness
__device__ void update_best(const State& state, const uint& g_idx) {
    if (state.fitness[g_idx] < state.best_fitness[g_idx]) {
        state.best_fitness[g_idx] = state.fitness[g_idx];
        for (uint i = 0; i < state.dim; i++) {
            state.best_position[g_idx * state.dim + i] = state.position[g_idx * state.dim + i];
        }
    }
}


// Update neighborhood best historical position and fitness (star topology)
// TODO add new topologies
__device__ void share(const State& state, const uint& g_idx) {
    uint p_idx = (g_idx > 0 ? g_idx : state.n) - 1;
    uint n_idx = g_idx < state.n - 1 ? g_idx + 1 : 0;
    uint b_idx = state.fitness[g_idx] < state.fitness[p_idx] ? g_idx : p_idx;
    b_idx = state.fitness[b_idx] < state.fitness[n_idx] ? b_idx : n_idx;
    if (state.fitness[b_idx] < state.best_known_fitness[g_idx]) {
        state.best_known_fitness[g_idx] = state.fitness[b_idx];
        for (uint i = 0; i < state.dim; i++) {
            state.best_known_position[g_idx * state.dim + i] = state.position[b_idx * state.dim + i];
        }
    }
}


// Update current position and velocity
__device__ void update_particle_state(const State& state, const uint& g_idx, rocrand_state_xorwow& rng_state) {
    state.r_p[g_idx] = rocrand_uniform(&rng_state);
    state.r_g[g_idx] = rocrand_uniform(&rng_state);
    for (uint i = 0; i < state.dim; i++) {
        state.velocity[g_idx * state.dim + i] =
            (state.momentum * state.velocity[g_idx * state.dim + i]) +

            (state.cognitive_coefficient * state.r_p[g_idx] *
             (state.best_position[g_idx * state.dim + i] - state.position[g_idx * state.dim + i])) +

            (state.social_coefficient * state.r_g[g_idx] *
             (state.best_known_position[g_idx * state.dim + i] - state.position[g_idx * state.dim + i]));

        state.position[g_idx * state.dim + i] += state.velocity[g_idx * state.dim + i];
    }
}



// Main PSO kernel, initilize particles and run PSO over specified iterations.
// The kernel do the work of index checking, synchronization and shared_memory data transfers (TODO) for the device
// functions (maybe not for `share` as requirements are topology dependant ?? TODO).
template <EvalFunction EF>
__global__ void pso_kernel(const State* state_ptr) {
    State state = *state_ptr;
    thread_group gg = this_grid();

    uint g_idx = gg.thread_rank();

    rocrand_state_xorwow rng_state;
    rocrand_init(state.seed, g_idx, 0, &rng_state);

    const bool participates = g_idx < state.n;  // whether thread participates in some computations

    if (!state.initialized) {
        if (participates) init(state, g_idx, rng_state);
        if (participates) compute_fitness<EF>(state, g_idx);
        if (participates) update_best(state, g_idx);
        gg.sync();
        if (participates) share(state, g_idx);
        gg.sync();
    }

    for (uint i = 0; i < state.iterations; i++) {
        if (participates) update_particle_state(state, g_idx, rng_state);
        if (participates) compute_fitness<EF>(state, g_idx);
        if (participates) update_best(state, g_idx);
        gg.sync();
        if (participates) share(state, g_idx);
        gg.sync();
    }
}

// Kernel retrieving best position and fitness ever reached
__global__ void best_position_and_fitness(
    const State* const state_ptr, float* const best_fitness, float* const best_position, uint* const idxs
) {
    uint g_idx = this_grid().thread_rank();
    // uint t_idx = this_thread_block().thread_rank();
    // uint w_idx = t_idx / WARP_SIZE;
    // uint l_idx = t_idx % WARP_SIZE;

    State state = *state_ptr;

    uint running_threads = std::bit_floor(state.n);

    if (g_idx < running_threads) {
        if (2 * g_idx >= state.n) {
            idxs[g_idx] = g_idx;
        } else {
            idxs[g_idx] = state.best_fitness[g_idx] < state.best_fitness[2 * g_idx] ? g_idx : 2 * g_idx;
        }
    }
    running_threads >>= 1;

    sync(this_grid());

    for (; running_threads > 1; running_threads >>= 1) {
        if (g_idx < running_threads) {
            idxs[g_idx] =
                state.best_fitness[idxs[g_idx]] <= state.best_fitness[idxs[2 * g_idx]] ? idxs[g_idx] : idxs[2 * g_idx];
        }
        sync(this_grid());
    }

    if (g_idx == 0) {
        *best_fitness = state.best_fitness[0];
    }

    if (g_idx < state.dim) {
        for (uint i = 0; i < state.dim; i++) {
            best_position[g_idx] = state.best_position[idxs[0] + g_idx];
        }
    }
}


#else
static_assert(false, "`MAX_THREADS_PER_BLOCK` not defined");
#endif

#else
static_assert(false, "`WARP_SIZE` not defined");
#endif

#endif
