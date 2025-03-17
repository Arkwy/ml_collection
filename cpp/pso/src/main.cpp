#include <iostream>

#include "pso/pso.hpp" // algorithm logic
#include "pso/topology.hpp" // list and implementations of the different topologies
#include "pso/eval_function.hpp" // 
#include "pso/space.hpp" // definition space


// The function to run PSO on can be defined by template specialization.
using MyEvalFunction = EvalFunction<BoxSpace<2>>;

template <>
__device__ float MyEvalFunction::eval_point(const float* const point) {
    return (point[0] - 1) * (point[0] - 1) + point[1] * point[1];
}

// Or by inheriting `EvalFunction` and using CRTP, which allows compiling with multiple different functions to evaluate.
struct MyDerivedEvalFunction
    : public EvalFunction<BoxSpace<2>, PointEvaluationMode::SingleThreaded, MyDerivedEvalFunction> {
    __device__ static float eval_point(const float* const point) { return point[0] + 2.0 * point[1]; }
};


struct MyOtherDerivedEvalFunction
    : public EvalFunction<BoxSpace<2>, PointEvaluationMode::SingleThreaded, MyOtherDerivedEvalFunction> {
    __device__ static float eval_point(const float* const point) { return point[0] - 2.0 * point[1]; }
};




// // For high dimension space, you can use multiple GPU threads to compute the fitness of each point
// // Use 
// using MyHighDimensionEvalFunction = EvalFunction<BoxSpace<100>, PointEvaluationMode::MultiThreaded>;

// template <>
// __device__ float MyHighDimensionEvalFunction::eval_point(const float* const point, const uint dim, float* const result) {
//     __shared__ 
// }

int main() {
    const uint N = 10;

    MyEvalFunction ef(BoxSpace<2>({{{-10, 10}, {-10, 10}}}));

    PSO<N, MyEvalFunction, Topology<TopologyCategory::RING>> pso(ef, 0.5, 0.4, 0.6);

    pso.step();     // perform a single iteration
    pso.run(1000);  // perform 100 iterations

    // get the point with lower fitness found by the algorithm
    auto [best_pos, best_fitness] = pso.best();

    std::cout << best_pos << " => " << best_fitness << std::endl;

    return 0;
}
