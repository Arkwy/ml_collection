#ifndef PSO_H
#define PSO_H

#include <sstream>
#include <stdexcept>

#include "../array/nd_array.hpp"
#include "particles.hpp"

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


template <size_t N, size_t D, typename Space>
struct PSO {
	const Space space;
	const Particles<N, D> particles;
	const Topology topology;
	const NDArray<uint32_t, N, N> connections;
	const float weight;
	const float cognitive_coefficient;
	const float social_coefficient;
	PSO(const Space& space,
		const Topology& topology,
		const float& weight,
		const float& cognitive_coefficient,
		const float& social_coefficient)
		: space(space),
		  particles(),
		  topology(topology),
		  connections(),
		  weight(weight),
		  cognitive_coefficient(cognitive_coefficient),
		  social_coefficient(social_coefficient) {
		space.sample(particles.positions);
        LOG(LOG_LEVEL_INFO, "pos ok");
		space.sample(particles.velocities);
        LOG(LOG_LEVEL_INFO, "vel ok");
		init_connections();
	}

  private:
	void init_connections() {
		switch (topology) {
			case STAR:
				// connections = true;
				break;
			default:
				throw std::invalid_argument((std::ostringstream()
											 << "PSO topology " << topology_to_string(topology) << " not implemented.")
												.str()
												.c_str());
		}
	}
};


#endif
