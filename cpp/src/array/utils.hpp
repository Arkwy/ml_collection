#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

template <size_t... N>
struct mul;

template <size_t N>
struct mul<N> {
    constexpr static const size_t value = N;
};

template <size_t N, size_t... M>
struct mul<N, M...> {
    constexpr static const size_t value = N * mul<M...>::value;
};

#endif
