#ifndef EVAL_FUNCTION_H
#define EVAL_FUNCTION_H

// Expected user defined struct that should contain
// - a static member `dim` specifying spatial dimensionality of the evaluation function
// - a static __device__ function `eval` returning the result of applying the function to a given point
template <typename T>
concept EvalFunction = requires {
    { T::dim } -> std::convertible_to<size_t>;
    requires(T::dim > 0);
// } && requires(const float (&point)[T::dim]) { // TODO uptate to use this signature
} && requires(const float *point) {
    { T::eval(point) } -> std::same_as<float>;
} && requires {
    [] __device__() {  // Checks if `T::eval` is callable in a __device__ context
        float test[T::dim] = {0};
        T::eval(test);
    };
};

#endif
