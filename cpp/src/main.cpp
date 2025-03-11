#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <iostream>
#include <ranges>
#include <type_traits>

#include "pso/pso.hpp"
#include "pso/space.hpp"
#include "pso/eval_function.hpp"

// static constexpr PointEvaluationMode ST = PointEvaluationMode::SingleThreaded;

// struct F {
//     using PEM = PointEvaluationMode::SingleThreaded;
// 	__device__ static float eval_point(const float* const point) { return point[0] + 2 * point[1] - point[2]; }
// };

// struct MySpace : public BoxSpace<3, PointEvaluationMode::SingleThreaded> {
// 	MySpace(const std::array<std::array<float, 2>, 3>& bounds) : BoxSpace<3, PointEvaluationMode::SingleThreaded>(bounds) {}
// 	__device__ static float eval_point(const float* const point) {
//         return 1;
// 	}
// };


// struct MyEvalFunction : EvalFunction<PointEvaluationMode::SingleThreaded, BoxSpace<3>> {
// 	__global__ static void eval_points(const float* const points, float* const result, int N) {
//         int point_idx = threadIdx.x + blockDim.x * blockIdx.x;
//         if (point_idx < N) {
//             result[point_idx] = eval_point(points + point_idx * Sc::dim);
//         }
//     }

// 	__device__ static float eval_point(const float* const point) {
//         return 1;
// 	}
// };

using MyEvalFunction = EvalFunction<PointEvaluationMode::SingleThreaded, BoxSpace<3>>;

template <>
__device__ float MyEvalFunction::eval_point(const float* const point) {
    return 1;
}

int main() {
	const size_t N = 10;

    MyEvalFunction ef(BoxSpace<3>({{{-1, 1}, {-1, 1}, {-1, 1}}}));
	Particles<N, 3> particles;
	ef.space.sample(particles.positions);
	ef.space.sample(particles.velocities);
    std::cout << ef.space.bounds << std::endl;
	std::cout << particles << std::endl;

    NDArray<float, N> fitness;
    ef.eval_points<<<N, 1>>>(particles.positions.get_device(), fitness.get_mut_device(), N);

	std::cout << fitness << std::endl;
	std::cout << particles << std::endl;

    // using MySpace = BoxSpace<3, PointEvaluationMode::SingleThreaded>;

	// MySpace space({{{-10, 10}, {-10, 10}, {-10, 10}}});


	// PSO<N, MySpace, Topology::STAR> pso(space, 0.5, 0.5, 0.5);

    // MySpace::eval_points<<<10, 1>>>(pso.particles.positions.get_device(), pso.fitness.get_mut_device(), 10);
	// // std::cout << pso.particles << std::endl;

	// // NDArray<float, D, 2> boundaries(std::array<float, D*2>({-10., 10., -100., 100.}));
	// // Particles<N, D> particles(boundaries);
	// // particles.sync();
	// std::cout << pso.particles << std::endl;
	// std::cout << pso.fitness << std::endl;

	return 0;
}
