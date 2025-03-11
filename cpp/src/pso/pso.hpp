#ifndef PSO_H
#define PSO_H

#include <hip/amd_detail/amd_hip_runtime.h>

#include <cstdint>

#include "../array/nd_array.hpp"
#include "particles.hpp"
#include "eval_function.hpp"


enum Topology {
	STAR,
};

inline const char* topology_to_string(int topology) {
	switch (topology) {
		case STAR:
			return "STAR";
			break;
		default:
			return "UNKNOWN";
	}
}

// Trait to check if a class is derived from a Space<D> whatever is D
template <typename T>
struct is_eval_function {
	static constexpr bool value = decltype(check(std::declval<T*>()))::value;

  private:
	template <PointEvaluationMode PEM, SpaceType S>
	static std::true_type check(EvalFunction<PEM, S>*);	 // Matches if T inherits Space<D> for any size_t D

	static std::false_type check(...);	// Maches any T if first overload doesn't match
};

template <typename T>
concept EvalFunctionType = inherit_space<T>::value;

template <size_t NumParticles, EvalFunctionType EF, Topology Topology_>
struct PSO {
	const EF eval_function;
	const float momentum;
	const float cognitive_coefficient;
	const float social_coefficient;

	const Particles<NumParticles, EF::Space::dim> particles;
	const NDArray<float, NumParticles> fitness;
	const NDArray<uint32_t, (NumParticles + 31) / 32, (NumParticles + 31) / 32> neighborhood;  // bit mask
	const NDArray<float, NumParticles> particle_best_fitness;
	const NDArray<float, EF::Space::dim, NumParticles> particle_best_position;
	const NDArray<float, NumParticles> particle_best_known_fitness;
	const NDArray<float, EF::Space::dim, NumParticles> particle_best_known_position;

	PSO(const EF& eval_function, const float& weight, const float& cognitive_coefficient, const float& social_coefficient)
		: eval_function(eval_function),
		  momentum(weight),
		  cognitive_coefficient(cognitive_coefficient),
		  social_coefficient(social_coefficient),
		  particles(),
		  fitness(),
		  neighborhood(),
		  particle_best_fitness(),
		  particle_best_position(),
		  particle_best_known_fitness(),
		  particle_best_known_position() {
		eval_function.space.sample(particles.positions);
		eval_function.space.sample(particles.velocities);
		init_connections();
		init_scores();
	}

  private:
	void init_connections() {
		if constexpr (Topology_ == Topology::STAR) {
			neighborhood = 0xffffffff;
		} else {
			static_assert(false, "Unexpected Topology!");
		}
	}


	static __global__ void share_scores(const float* const connections, float* const best_known_score_per_particle) {}

	void init_scores() const {
		// Space::eval_points<<<NumParticles, 1>>>(
		// 	particles.positions.get_device(),
		// 	fitness.get_mut_device(),
        //     NumParticles
		// );
		// // particle_best_fitness.slice(0, Space::dim) = particles.positions;						// TODO
		// particle_best_fitness.slice(Space::dim, Space::dim + 1) = fitness;						// TODO
		// particle_best_known_fitness = particle_best_fitness;									// TODO
		// share_scores(neighborhood.get_device(), particle_best_known_fitness.get_mut_device());	// TODO
	}

	// def init_scores(self) -> None:
	//     self.scores = self.space(self.positions)

	//     g_scores = self.scores[None, :].expand_as(self.neighbors).where(
	//         self.neighbors,
	//         torch.inf,
	//     )

	//     bk_scores_id = g_scores.argmin(-1)
	//     self.bk_scores = self.scores[bk_scores_id]
	//     self.bk_positions = self.positions[bk_scores_id]

	//     best_score_id = self.scores.argmin()
	//     self.best_score = self.scores[best_score_id]
	//     self.best_position = self.positions[best_score_id]
};


#endif
