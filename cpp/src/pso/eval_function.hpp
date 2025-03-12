#ifndef EVAL_FUNCTION_H
#define EVAL_FUNCTION_H

#include <cstddef>
#include <cstdio>
#include <type_traits>

#include "space.hpp"

enum class PointEvaluationMode { SingleThreaded, MultiThreaded, Custom };


// Trait to check if a class is derived from a Space for any D
template <typename T>
class inherit_space {
	template <size_t D>
	static std::true_type check(Space<D>*);	 // Matches if T inherits Space<D>.

	static std::false_type check(...);	// Matches anything that don't match first overlaod of `check`.

  public:
	static constexpr bool value = decltype(check(std::declval<T*>()))::value;
};


template <typename T>
concept SpaceType = inherit_space<T>::value;


template <PointEvaluationMode PEM>
concept is_single_threaded = std::integral_constant<bool, PEM == PointEvaluationMode::SingleThreaded>::value;


template <PointEvaluationMode PEM>
concept is_multi_threaded = std::integral_constant<bool, PEM == PointEvaluationMode::MultiThreaded>::value;


template <PointEvaluationMode PEM, SpaceType S>
struct EvalFunction {
	using DefinitionSpace = S;
	const DefinitionSpace space;  // PSO get access to space through EvalFunction

	EvalFunction(const DefinitionSpace& space) : space(space) {}

	__device__ static float eval_point(const float* const point) {
		if constexpr (!is_single_threaded<PEM>) {
			static_assert(
				false,
				"Call to single threaded `eval_point` but PointEvaluationMode is not `SingleThreaded`."
			);
		} else {
			static_assert(false, "You must implement `eval_point` through template specialization.");
		}
	}

	__device__ static void eval_point(const float* const point, const int dim_idx, float& result) {
		if constexpr (!is_multi_threaded<PEM>) {
			static_assert(false, "Call to multi threaded `eval_point` but PointEvaluationMode is not `MultiThreaded`.");
		} else {
			static_assert(false, "You must implement `eval_point` through template specialization.");
		}
	}

	__global__ static void eval_points(const float* const points, float* const result, const int N)
		requires is_single_threaded<PEM>
	{
		int point_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (point_idx < N) {
			result[point_idx] = eval_point(points + point_idx * DefinitionSpace::dim);
		}
	}

	__global__ static void eval_points(const float* const points, float* const result, const int N)
		requires is_multi_threaded<PEM>
	{
		int point_idx = threadIdx.y + blockDim.y * blockIdx.y;
		int dim_idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (point_idx < N && dim_idx < DefinitionSpace::dim) {
			eval_point(points + point_idx * DefinitionSpace::dim, dim_idx, result[point_idx]);
		}
	}
};

#endif
