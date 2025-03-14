#ifndef OPS_REDUCE_OPS_H
#define OPS_REDUCE_OPS_H

#include <hip/hip_runtime.h>

#include <type_traits>


template <typename T>
concept OpSupportedType =
    std::same_as<T, double> || std::same_as<T, float> || std::same_as<T, int32_t> || std::same_as<T, uint32_t>;

template <OpSupportedType T>
struct Add {
    using IdOp = std::false_type;
    using DType = T;
    static constexpr T identity = 0;
    __device__ static T run(const T& a, const T& b) { return a + b; }
};


template <OpSupportedType T>
struct Mul {
    using IdOp = std::false_type;
    using DType = T;
    static constexpr T identity = 1;
    __device__ static T run(const T& a, const T& b) { return a * b; }
};


template <OpSupportedType T>
struct Min {
    using IdOp = std::false_type;
    using DType = T;
    static constexpr T identity = std::numeric_limits<T>::infinity();
    __device__ static T run(const T& a, const T& b) { return a < b ? a : b; }
};


template <OpSupportedType T>
struct Max {
    using IdOp = std::false_type;
    using DType = T;
    static constexpr T identity = -std::numeric_limits<T>::infinity();
    __device__ static T run(const T& a, const T& b) { return a < b ? b : a; }
};


template <OpSupportedType T>
struct ArgMin {
    using IdOp = std::true_type;
    using DType = T;
    static constexpr T identity = std::numeric_limits<T>::infinity();
    __device__ static void run(T& a, const T& b, uint& a_idx, const uint& b_idx) {
        if (a > b) {
            a = b;
            a_idx = b_idx;
        }
    }
};


template <OpSupportedType T>
struct ArgMax {
    using IdOp = std::true_type;
    using DType = T;
    static constexpr T identity = -std::numeric_limits<T>::infinity();
    __device__ static void run(T& a, const T& b, uint& a_idx, const uint& b_idx) {
        if (a < b) {
            a = b;
            a_idx = b_idx;
        }
    }
};


template <typename T>
concept is_no_id_op = requires {
    typename T::IdOp;
    typename T::DType;
} && OpSupportedType<typename T::DType> && requires {
    !T::IdOp::value;
} && requires(const typename T::DType& a, const typename T::DType& b) {
    { T::run(a, b) } -> std::same_as<typename T::DType>;
} && requires {
    { T::identity } -> std::convertible_to<typename T::DType>;
};


template <typename T>
concept is_id_op = requires {
    typename T::IdOp;
    typename T::DType;
} && OpSupportedType<typename T::DType> && requires {
    T::IdOp::value;
} && requires(typename T::DType& a, const typename T::DType& b, uint& a_idx, const uint& b_idx) {
    { T::run(a, b, a_idx, b_idx) } -> std::same_as<void>;
} && requires {
    { T::identity } -> std::convertible_to<typename T::DType>;
};

#endif
