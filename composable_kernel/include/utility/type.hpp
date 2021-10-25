#ifndef CK_TYPE_HPP
#define CK_TYPE_HPP

#include <cstring>
#include "integral_constant.hpp"
#include "enable_if.hpp"

namespace ck {

template <typename X, typename Y>
struct is_same : public integral_constant<bool, false>
{
};

template <typename X>
struct is_same<X, X> : public integral_constant<bool, true>
{
};

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
inline constexpr bool is_pointer_v = std::is_pointer<T>::value;

template <typename T>
struct is_known_at_compile_time;

template <>
struct is_known_at_compile_time<index_t>
{
    static constexpr bool value = false;
};

template <typename T, T X>
struct is_known_at_compile_time<integral_constant<T, X>>
{
    static constexpr bool value = true;
};

// https://stackoverflow.com/questions/44137442/what-is-type-punning-and-what-is-the-purpose-of-it
// https://en.cppreference.com/w/cpp/string/byte/memcpy
// https://clang.llvm.org/docs/LanguageExtensions.html
template <typename Y, typename X, typename enable_if<sizeof(X) == sizeof(Y), bool>::type = false>
__host__ __device__ constexpr Y as_type(X x)
{
#if 0 // debug
    union AsType
    {
        X x;
        Y y;
    };

    return AsType{x}.y;
#else
    Y y;

    __builtin_memcpy_inline(&y, &x, sizeof(X));

    return y;
#endif
}

} // namespace ck
#endif
