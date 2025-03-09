#include <hip/hip_runtime.h>

#include <cstdio>
#include <iostream>

#include "pso/pso.hpp"
#include "pso/space.hpp"


int main() {
	const size_t D = 32;
	const size_t N = 1024;

    BoxSpace<D> space({{{-10, 10}, {-10, 10}, {-100, 100}}});

	PSO<N, D, BoxSpace<D>> pso(space, Topology::STAR, 0.5, 0.5, 0.5);

	std::cout << pso.particles << std::endl;

	// NDArray<float, D, 2> boundaries(std::array<float, D*2>({-10., 10., -100., 100.}));
	// Particles<N, D> particles(boundaries);
	// particles.sync();
	// std::cout << particles << std::endl;

	return 0;
}
