// #include <hip/amd_detail/amd_hip_runtime.h>
// #include <hip/hip_runtime.h>

// #include <cstdio>
// #include <iostream>

// #include "pso/eval_function.hpp"
// #include "pso/pso.hpp"
// #include "pso/space.hpp"

// using MyEvalFunction = EvalFunction<PointEvaluationMode::SingleThreaded, BoxSpace<3>>;

// template <>
// __device__ float MyEvalFunction::eval_point(const float* const point) {
//     return 1.23;
// }

// int main() {
// 	const size_t N = 1025;

// 	MyEvalFunction ef(BoxSpace<3>({{{-1, 1}, {-1, 1}, {-1, 1}}}));

//     PSO<N, MyEvalFunction, Topology::STAR> pso(ef, 0.5, 0.4, 0.6);

//     std::cout << pso.fitness << std::endl;
//     std::cout << pso.particle_best_fitness << std::endl;

// 	return 0;
// }

#include "ops/reduce/1d.hpp"

int main() {
	constexpr uint N = 2000;

	NDArray<double, N> data;
	for (int i = 0; i < N; i++) {
		data[i] = (double) -i;
	}

	int max = reduce_1d<Max<double>, N>(data);
	std::cout << max << std::endl;

	int min = reduce_1d<Min<double>, N>(data);
	std::cout << min << std::endl;

	int arg_max = reduce_1d<ArgMax<double>, N>(data);
	std::cout << arg_max << std::endl;

	int arg_min = reduce_1d<ArgMin<double>, N>(data);
	std::cout << arg_min << std::endl;

	return 0;
}
