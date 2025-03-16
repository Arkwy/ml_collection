#include <iostream>

#include "pso/eval_function.hpp"
#include "pso/pso.hpp"
#include "pso/space.hpp"

using MyEvalFunction = EvalFunction<PointEvaluationMode::SingleThreaded, BoxSpace<3>>;

template <>
__device__ float MyEvalFunction::eval_point(const float* const point) {
    return point[0] + 2.0 * point[1] - point[2];
}

int main() {
	const size_t N = 10;

	MyEvalFunction ef(BoxSpace<3>({{{-1, 1}, {-1, 1}, {-1, 1}}}));

    PSO<N, MyEvalFunction, Topology::GLOBAL> pso(ef, 0.5, 0.4, 0.6);

    std::cout << pso.particles.position << std::endl;
    std::cout << pso.particles.fitness << std::endl;

    // pso.run(100); // TODO

	return 0;
}
