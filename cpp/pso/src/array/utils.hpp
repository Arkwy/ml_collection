#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

template <uint... N>
struct mul;

template <uint N>
struct mul<N> {
    constexpr static const uint value = N;
};

template <uint N, uint... M>
struct mul<N, M...> {
    constexpr static const uint value = N * mul<M...>::value;
};

#endif
