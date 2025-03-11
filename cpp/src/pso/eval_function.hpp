#ifndef EVAL_FUNCTION_H
#define EVAL_FUNCTION_H

#include <cstddef>
#include <type_traits>

#include "space.hpp"

// template <size_t D>
// struct Space {};

enum class PointEvaluationMode { SingleThreaded, MultiThreaded, Custom };


// Trait to check if a class is derived from a template
template <typename Derived>
struct inherit_space {
  private:
	template <size_t D>
	static std::true_type check(Space<D>*);	 // Matches if Derived is Base<T>

	static std::false_type check(...);	// Fallback

  public:
	static constexpr bool value = decltype(check(std::declval<Derived*>()))::value;
};

template <typename T>
concept SpaceType = inherit_space<T>::value;

template <PointEvaluationMode PEM>
concept is_single_threaded = std::integral_constant<bool, PEM == PointEvaluationMode::SingleThreaded>::value;

template <PointEvaluationMode PEM>
concept is_multi_threaded = std::integral_constant<bool, PEM == PointEvaluationMode::MultiThreaded>::value;

template <PointEvaluationMode PEM, SpaceType S>
struct EvalFunction {
	using Sc = S;
	const Sc space;

	EvalFunction(const Sc& space) : space(space) {}

	__device__ static float eval_point(const float* const point) {
		if constexpr (!is_single_threaded<PEM>) {
            static_assert(false, "unallowed call to ..");
		} else {
            static_assert(false, "you sould provide your own implem");
		}
	}

	__device__ static void eval_point(const float* const point, const int dim_idx, float& result)
		requires is_multi_threaded<PEM>;

	__global__ static void eval_points(const float* const points, float* const result, int N)
		requires is_single_threaded<PEM>
	{
		int point_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (point_idx < N) {
			result[point_idx] = eval_point(points + point_idx * Sc::dim);
		}
	}
};

// template <PointEvaluationMode PEM, SpaceType S>
// __device__ float EvalFunction<PEM, S>::eval_point(const float* const point)
// 	requires std::integral_constant<bool, PEM == PointEvaluationMode::SingleThreaded>::value
// {
//     // static_assert(false, "implem not provided");
// }
#endif
