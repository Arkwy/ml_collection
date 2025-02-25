
// template <typename Derived, typename T, typename D, size_t N>
// TensorBase<Derived, T, D, N> &TensorBase<Derived, T, D, N>::reshape(const array<size_t, N> &new_shape) {
//     assert(numel_from_shape(new_shape) == numel());
//     if (!is_contiguous()) {
//         shape = new_shape;
//         stride = base_stride_from_shape(shape);
//         // offset already 0
//         return *this;
//     } else {
//         throw runtime_error("view flatten not implemented");
//     } // TODO provide implementation for views
// }
