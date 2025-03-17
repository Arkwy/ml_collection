#include <iostream>

#include "pso/eval_function.hpp"  //
#include "pso/pso.hpp"            // algorithm logic
#include "pso/space.hpp"          // definition space
#include "pso/topology.hpp"       // list and implementations of the different topologies


// The function to run PSO on can be defined by template specialization.
using MyEvalFunction = EvalFunction<BoxSpace<2>>;

template <>
__device__ float MyEvalFunction::eval_point(const float* const point) {
    // Do not use __syncthreads() or shared memory or issues may arise (you should not need them here anyway).
    return (point[0] - 1) * (point[0] - 1) + point[1] * point[1];
}


// Or by inheriting `EvalFunction` and using CRTP, which allows compiling with multiple different functions to evaluate.
// Note that in this case you must provide the `PointEvaluationMode` template argument which defaults to
// `SingleThreaded`. More on that bellow.
struct MyDerivedEvalFunction
    : public EvalFunction<BoxSpace<2>, PointEvaluationMode::SingleThreaded, MyDerivedEvalFunction> {
    __device__ static float eval_point(const float* const point) { return point[0] + 2.0 * point[1]; }
};


struct MyOtherDerivedEvalFunction
    : public EvalFunction<BoxSpace<2>, PointEvaluationMode::SingleThreaded, MyOtherDerivedEvalFunction> {
    __device__ static float eval_point(const float* const point) { return point[0] - 2.0 * point[1]; }
};



// For high dimension space, you can use multiple GPU threads to compute the fitness of each point.
// This should support up to a 1024D space (depends on your GPU). Above 1024 , use `PointEvaluationMode::Custom` (not
// actually implemented yet :( ).
#define D 1024
using MyHighDimensionEvalFunction = EvalFunction<BoxSpace<D>, PointEvaluationMode::MultiThreaded>;

template <>
__device__ void MyHighDimensionEvalFunction::eval_point(
    const float* const point,  // start of the array representing the point to process
    const uint dim,            // dimension to process by this GPU thread
    float& result              // variable to write result (= fitness) in.
) {
    // eval function example: returns the euclidean distance to the origin

    __shared__ float s_point[D];  // use shared memory for faster access during computation
    if (dim < D) {                // don't forget to make sure dim is valid as it is not guaranted
        s_point[dim] = point[dim] * point[dim];
    }
    __syncthreads();

    // perform sum reduce (changing D for something else than a power of two requires adapting this loop)
    for (uint stride = D / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            s_point[dim] = s_point[dim] + s_point[dim + stride];
        }
        __syncthreads();
    }

    if (dim == 0) {  // make sure only one thread writes to result
        result = sqrt(s_point[0]);
    }
}



int main() {
    constexpr const uint N = 10000;  // TODO find why it's crash when N is too large (too many blocks launched ?)


    MyEvalFunction ef(BoxSpace<2>({{{-10, 10}, {-10, 10}}}));
    PSO<N, MyEvalFunction, Topology<TopologyCategory::RING>> pso(ef, 0.5, 0.4, 0.6);


    // std::array<std::array<float, 2>, D> bounds;
    // bounds.fill({-1, 1});
    // MyHighDimensionEvalFunction ef(bounds);
    // PSO<N, MyHighDimensionEvalFunction, Topology<TopologyCategory::RING>> pso(ef, 0.5, 0.4, 0.6);


    pso.step();   // perform a single iteration
    pso.run(10000);  // perform 1000 iterations

    // get the point with the lowest fitness found by the algorithm
    auto [best_pos, best_fitness] = pso.best();

    std::cout << best_pos << " => " << best_fitness << std::endl;

    return 0;
}
