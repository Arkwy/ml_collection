#include <iostream>

#include "pso/eval_function.hpp"
#include "pso/pso.hpp"
#include "pso/space.hpp"
#include "pso/topology.hpp"


// The function to run PSO on can be defined by template specialization.
using MyEvalFunction = EvalFunction<PointEvaluationMode::SingleThreaded, BoxSpace<3>>;

template <>
__device__ float MyEvalFunction::eval_point(const float* const point) {
    return 0;
}


// Or by inheriting `EvalFunction` and using CRTP, which allows compiling with multiple different functions to evaluate.
struct MyDerivedEvalFunction
    : public EvalFunction<PointEvaluationMode::SingleThreaded, BoxSpace<3>, MyDerivedEvalFunction> {
    __device__ static float eval_point(const float* const point) { return point[0] + 2.0 * point[1] - point[2]; }
};


struct MyOtherDerivedEvalFunction
    : public EvalFunction<PointEvaluationMode::SingleThreaded, BoxSpace<3>, MyOtherDerivedEvalFunction> {
    __device__ static float eval_point(const float* const point) { return point[0] - 2.0 * point[1] + point[2]; }
};



int main() {
    const size_t N = 10;

    MyEvalFunction ef(BoxSpace<3>({{{-1, 1}, {-1, 1}, {-1, 1}}}));

    PSO<N, MyEvalFunction, Topology<TopologyCategory::GLOBAL>> pso(ef, 0.5, 0.4, 0.6);

    std::cout << pso.particles.position << std::endl;
    std::cout << pso.particles.fitness << std::endl;


    MyDerivedEvalFunction d_ef(BoxSpace<3>({{{-1, 1}, {-1, 1}, {-1, 1}}}));

    PSO<N, MyDerivedEvalFunction, Topology<TopologyCategory::GLOBAL>> d_pso(d_ef, 0.5, 0.4, 0.6);

    std::cout << d_pso.particles.position << std::endl;
    std::cout << d_pso.particles.fitness << std::endl;


    MyOtherDerivedEvalFunction od_ef(BoxSpace<3>({{{-1, 1}, {-1, 1}, {-1, 1}}}));

    PSO<N, MyOtherDerivedEvalFunction, Topology<TopologyCategory::GLOBAL>> od_pso(od_ef, 0.5, 0.4, 0.6);

    std::cout << od_pso.particles.position << std::endl;
    std::cout << od_pso.particles.fitness << std::endl;
    // pso.run(100); // TODO

    return 0;
}
