#ifndef CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP
#define CK_DYNAMIC_MULTI_INDEX_TRANSFORM_HPP

#include "common_header.hpp"
#include "multi_index.hpp"

namespace ck {

template <typename LowLength>
struct DynamicPassThrough
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{}));

    UpLengths up_lengths_;

    __host__ __device__ constexpr DynamicPassThrough() = default;

    __host__ __device__ constexpr DynamicPassThrough(const LowLength& low_length)
        : up_lengths_{make_tuple(low_length)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx& idx_up_new,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("DynamicPassThrough, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename LowLength, typename LeftPad, typename RightPad, bool SkipIsValidCheck = false>
struct DynamicPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + LeftPad{} + RightPad{}));

    UpLengths up_lengths_;
    LeftPad left_pad_;
    RightPad right_pad_;

    __host__ __device__ constexpr DynamicPad() = default;

    __host__ __device__ constexpr DynamicPad(const LowLength& low_length,
                                             const LeftPad& left_pad,
                                             const RightPad& right_pad)
        : up_lengths_{make_tuple(low_length + left_pad + right_pad)},
          left_pad_{left_pad},
          right_pad_{right_pad}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] - left_pad_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx& idx_up_new,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || ((idx_up[Number<0>{}] >= left_pad_) &&
                                    (idx_up[Number<0>{}] < up_lengths_[Number<0>{}] - right_pad_));
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LeftPad>::value &&
               is_known_at_compile_time<RightPad>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("DynamicPad, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("left_pad_ %d", index_t{left_pad_});
        printf("right_pad_ %d", index_t{right_pad_});
        printf("}");
    }
};

template <typename LowLength, typename LeftPad, bool SkipIsValidCheck = false>
struct DynamicLeftPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + LeftPad{}));

    UpLengths up_lengths_;
    LeftPad left_pad_;

    __host__ __device__ constexpr DynamicLeftPad() = default;

    __host__ __device__ constexpr DynamicLeftPad(const LowLength& low_length,
                                                 const LeftPad& left_pad)
        : up_lengths_{make_tuple(low_length + left_pad)}, left_pad_{left_pad}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] - left_pad_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx& idx_up_new,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[Number<0>{}] >= left_pad_);
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LeftPad>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("DynamicLeftPad, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("left_pad_ %d", index_t{left_pad_});
        printf("}");
    }
};

template <typename LowLength, typename RightPad, bool SkipIsValidCheck = false>
struct DynamicRightPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + RightPad{}));

    UpLengths up_lengths_;
    LowLength low_length_;
    RightPad right_pad_;

    __host__ __device__ constexpr DynamicRightPad() = default;

    __host__ __device__ constexpr DynamicRightPad(const LowLength& low_length,
                                                  const RightPad& right_pad)
        : up_lengths_{make_tuple(low_length + right_pad)},
          low_length_{low_length},
          right_pad_{right_pad}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                                  const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx& idx_up_new,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[Number<0>{}] < low_length_);
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LowLength>::value &&
               is_known_at_compile_time<RightPad>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("DynamicRightPad, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("low_length_ %d", index_t{low_length_});
        printf("left_pad_ %d", index_t{right_pad_});
        printf("}");
    }
};

// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1]
// UpLengths and Coefficients can be either of the followings:
//   1) Tuple of index_t, which is known at run-time, or
//   2) Tuple of Number, which is known at compile-time, or
//   3) Tuple of mixture of index_t and Number, which is known partially at run-time and partially
//   at compile-time
template <typename UpLengths,
          typename Coefficients,
          typename std::enable_if<UpLengths::Size() == Coefficients::Size(), bool>::type = false>
struct DynamicEmbed
{
    static constexpr index_t NDimUp = UpLengths::Size();

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    UpLengths up_lengths_;
    Coefficients coefficients_;

    __host__ __device__ constexpr DynamicEmbed() = default;

    __host__ __device__ constexpr DynamicEmbed(const UpLengths& up_lengths,
                                               const Coefficients& coefficients)
        : up_lengths_{up_lengths}, coefficients_{coefficients}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}([&idx_low, &idx_up, this](auto i) {
            idx_low(Number<0>{}) += idx_up[i] * this->coefficients_[i];
        });
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == NDimUp &&
                          LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}(
            [&](auto i) { idx_diff_low(Number<0>{}) += idx_diff_up[i] * coefficients_[i]; });

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<Coefficients>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("DynamicEmbed, ");
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("coefficients_ ");
        print_multi_index(coefficients_);
        printf("}");
    }
};

#if 1
template <typename LowLengths>
struct lambda_merge_generate_magic_division_calculate_magic_multiplier
{
    template <index_t I>
    __host__ __device__ constexpr auto operator()(Number<I> i) const
    {
        return magic_division::CalculateMagicMultiplier(LowLengths{}[i]);
    }
};

template <typename LowLengths>
struct lambda_merge_generate_magic_division_calculate_magic_shift
{
    template <index_t I>
    __host__ __device__ constexpr auto operator()(Number<I> i) const
    {
        return magic_division::CalculateMagicShift(LowLengths{}[i]);
    }
};

template <typename LowLengths>
struct DynamicMerge
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using LowLengthsScan = decltype(
        container_reverse_exclusive_scan(LowLengths{}, math::multiplies_v2{}, Number<1>{}));

    using LowLengthsMagicDivisorMultipiler = decltype(generate_tuple(
        lambda_merge_generate_magic_division_calculate_magic_multiplier<LowLengths>{},
        Number<NDimLow>{}));

    using LowLengthsMagicDivisorShift = decltype(
        generate_tuple(lambda_merge_generate_magic_division_calculate_magic_shift<LowLengths>{},
                       Number<NDimLow>{}));

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies_v2{}, Number<1>{})));

    LowLengths low_lengths_;
    LowLengthsScan low_lengths_scan_;
    LowLengthsMagicDivisorMultipiler low_lengths_magic_divisor_multiplier_;
    LowLengthsMagicDivisorShift low_lengths_magic_divisor_shift_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr DynamicMerge() = default;

    __host__ __device__ constexpr DynamicMerge(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{
              container_reverse_exclusive_scan(low_lengths, math::multiplies_v2{}, Number<1>{})},
          low_lengths_magic_divisor_multiplier_{generate_tuple(
              [&](auto i) { return magic_division::CalculateMagicMultiplier(low_lengths[i]); },
              Number<NDimLow>{})},
          low_lengths_magic_divisor_shift_{generate_tuple(
              [&](auto i) { return magic_division::CalculateMagicShift(low_lengths[i]); },
              Number<NDimLow>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies_v2{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[Number<0>{}];

#if 0 
        // normal division
        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_low(i) = tmp / this->low_lengths_scan_[i];
            tmp -= idx_low[i] * this->low_lengths_scan_[i];
        });

        idx_low(Number<NDimLow - 1>{}) = tmp;
#else
        // magic division
        static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
            index_t tmp2 =
                magic_division::DoMagicDivision(tmp,
                                                this->low_lengths_magic_divisor_multiplier_[i],
                                                this->low_lengths_magic_divisor_shift_[i]);
            idx_low(i) = tmp - tmp2 * this->low_lengths_[i];
            tmp        = tmp2;
        });

        idx_low(Number<0>{}) = tmp;
#endif
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex_1a(LowIdxDiff& idx_diff_low,
                                                 const UpIdxDiff& idx_diff_up,
                                                 LowIdx& idx_low,
                                                 const UpIdx& /* idx_up_new */,
                                                 Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        // CalculateLowerIndex(idx_diff_low_const) has multiple integer divisions.
        // However,
        //   1) If idx_diff_up is known at compile-time, then idx_diff_low_const
        //   can be calculated at compile-time.
        //   2) If idx_diff_up is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then
        //   idx_diff_low_const also
        //   doesn't change during the whole kernel execution. Compiler generated
        //   ISA should
        //   only caclculate idx_diff_low_const once and save it durinng the whole
        //   kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be
        // computed at
        //   run-time each time this function is called, and can be very expensive.
        LowerIndex idx_diff_low_const;
        LowerIndex idx_low_length_minus_idx_diff_low_const;
        LowerIndex idx_low_length_plus_idx_diff_low_const;

#if !CK_HACK_DYNAMIC_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
        index_t tmp = idx_diff_up[Number<0>{}];

#if 1
        // normal division
        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = tmp / low_lengths_scan_[i];
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = tmp;
#else
        // magic division
        static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
            index_t tmp2 =
                magic_division::DoMagicDivision(tmp,
                                                this->low_lengths_magic_divisor_multiplier_[i],
                                                this->low_lengths_magic_divisor_shift_[i]);
            idx_diff_low_const(i) = tmp - tmp2 * this->low_lengths_[i];
            tmp                   = tmp2;
        });

        idx_diff_low_const(Number<0>{}) = tmp;
#endif

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) = low_lengths_[i] - idx_diff_low_const[i];

            idx_low_length_plus_idx_diff_low_const(i) = low_lengths_[i] + idx_diff_low_const[i];
        });
#else
        // Hack: this force result into SGPR. Need to make sure the result is thread invariant
        index_t tmp = idx_diff_up[Number<0>{}];

#if 1
        // normal division
        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = __builtin_amdgcn_readfirstlane(tmp / low_lengths_scan_[i]);
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });
#else
        // magic division
        static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
            index_t tmp2 =
                magic_division::DoMagicDivision(tmp,
                                                this->low_lengths_magic_divisor_multiplier_[i],
                                                this->low_lengths_magic_divisor_shift_[i]);

            idx_diff_low_const(i) =
                __builtin_amdgcn_readfirstlane(tmp - tmp2 * this->low_lengths_[i]);
            tmp = tmp2;
        });
#endif

        idx_diff_low_const(Number<NDimLow - 1>{}) = __builtin_amdgcn_readfirstlane(tmp);

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) =
                __builtin_amdgcn_readfirstlane(low_lengths_[i] - idx_diff_low_const[i]);

            idx_low_length_plus_idx_diff_low_const(i) =
                __builtin_amdgcn_readfirstlane(low_lengths_[i] + idx_diff_low_const[i]);
        });
#endif

        if constexpr(Hack == 1)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t idx_low_tmp = idx_low[i] + carry;

                bool do_carry = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
        else if constexpr(Hack == 2)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t borrow = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t idx_low_tmp = idx_low[i] - borrow;

                bool do_borrow = idx_low_tmp < -idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) -= borrow;

                borrow = do_borrow ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] - borrow;

            idx_low += idx_diff_low;
        }
        else
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t idx_low_tmp = idx_low[i] + carry;

                bool do_carry  = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];
                bool do_borrow = idx_low_tmp < -idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];
                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
                carry = do_borrow ? -1 : carry;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex_1b(LowIdxDiff& idx_diff_low,
                                                 const UpIdxDiff& idx_diff_up,
                                                 LowIdx& idx_low,
                                                 const UpIdx& /* idx_up_new */,
                                                 Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        // CalculateLowerIndex(idx_diff_low_const) has multiple integer divisions.
        // However,
        //   1) If idx_diff_up is known at compile-time, then idx_diff_low_const
        //   can be calculated at compile-time.
        //   2) If idx_diff_up is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then
        //   idx_diff_low_const also
        //   doesn't change during the whole kernel execution. Compiler generated
        //   ISA should
        //   only caclculate idx_diff_low_const once and save it durinng the whole
        //   kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be
        // computed at
        //   run-time each time this function is called, and can be very expensive.
        LowerIndex idx_diff_low_const;
        LowerIndex idx_low_length_minus_idx_diff_low_const;
        LowerIndex idx_low_length_plus_idx_diff_low_const;

#if !CK_HACK_DYNAMIC_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
        index_t tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = tmp / low_lengths_scan_[i];
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = tmp;

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) = low_lengths_[i] - idx_diff_low_const[i];

            idx_low_length_plus_idx_diff_low_const(i) = low_lengths_[i] + idx_diff_low_const[i];
        });
#else
        // Hack: this force result into SGPR. Need to make sure the result is thread invariant
        index_t tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = __builtin_amdgcn_readfirstlane(tmp / low_lengths_scan_[i]);
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = __builtin_amdgcn_readfirstlane(tmp);

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) =
                __builtin_amdgcn_readfirstlane(low_lengths_[i] - idx_diff_low_const[i]);

            idx_low_length_plus_idx_diff_low_const(i) = low_lengths_[i] + idx_diff_low_const[i];
        });
#endif

        if constexpr(Hack == 1)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t idx_low_tmp = idx_low[i] + carry;

                bool do_carry = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
        else if constexpr(Hack == 2)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t borrow = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t negative_idx_low_tmp = borrow - idx_low[i];

                bool do_borrow = negative_idx_low_tmp > idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) -= borrow;

                borrow = do_borrow ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] - borrow;

            idx_low += idx_diff_low;
        }
        else
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t idx_low_tmp = idx_low[i] + carry;

                bool do_carry  = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];
                bool do_borrow = idx_low_tmp < -idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];
                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
                carry = do_borrow ? -1 : carry;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex_2(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff& idx_diff_up,
                                                LowIdx& idx_low,
                                                const UpIdx& /* idx_up_new */,
                                                Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        // CalculateLowerIndex(idx_diff_low_const) has multiple integer divisions.
        // However,
        //   1) If idx_diff_up is known at compile-time, then idx_diff_low_const
        //   can be calculated at compile-time.
        //   2) If idx_diff_up is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then
        //   idx_diff_low_const also
        //   doesn't change during the whole kernel execution. Compiler generated
        //   ISA should
        //   only caclculate idx_diff_low_const once and save it durinng the whole
        //   kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be
        //   computed at run-time each time this function is called, and can be
        //   very expensive.
        LowerIndex idx_diff_low_const;

#if !CK_HACK_DYNAMIC_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
        index_t tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = tmp / low_lengths_scan_[i];
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = tmp;
#else
        // Hack: this force result into SGPR. Need to make sure the result is thread invariant
        index_t tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = __builtin_amdgcn_readfirstlane(tmp / low_lengths_scan_[i]);
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = __builtin_amdgcn_readfirstlane(tmp);
#endif

        if constexpr(Hack == 1)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            bool do_carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                idx_diff_low(i) = idx_diff_low_const[i] + do_carry;

                index_t idx_low_tmp = idx_low[i] + idx_diff_low[i];

                do_carry = idx_low_tmp >= low_lengths_[i];

#if 0
                // TODO: use exec-mask inline asm, which use 1 VALU
                if(do_carry)
                {
                    idx_diff_low(i) -= low_lengths_[i];
                }
#elif 1
                // this use 2 VALU
                idx_diff_low(i) = do_carry ? idx_diff_low[i] - low_lengths_[i] : idx_diff_low[i];
#elif 1
                // this use 2 VALU
                index_t idx_diff_low_tmp = idx_diff_low[i] - low_lengths_[i];
                idx_diff_low(i)          = do_carry ? idx_diff_low_tmp : idx_diff_low[i];
#endif

                idx_low(i) += idx_diff_low[i];
            });

            constexpr auto I0 = Number<0>{};

            idx_diff_low(I0) = idx_diff_low_const[I0] + do_carry;

            idx_low(I0) += idx_diff_low[I0];
        }
        else if constexpr(Hack == 2)
        {
            // do borrow check on each low dimension in reversed order
            // do not need to check the first dimension
            bool do_borrow = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                idx_diff_low(i) = idx_diff_low_const[i] - do_borrow;

                index_t idx_low_tmp = idx_low[i] + idx_diff_low[i];

                do_borrow = idx_low_tmp < 0;

#if 0
                // TODO: use exec-mask inline asm
                if(do_borrow)
                {
                    idx_diff_low(i) += low_lengths_[i];
                }
#elif 1
                idx_diff_low(i) = do_borrow ? idx_diff_low[i] + low_lengths_[i] : idx_diff_low[i];
#elif 1
                index_t idx_diff_low_tmp = idx_diff_low[i] + low_lengths_[i];
                idx_diff_low(i)          = do_borrow ? idx_diff_low_tmp : idx_diff_low[i];
#endif

                idx_low(i) += idx_diff_low[i];
            });

            constexpr auto I0 = Number<0>{};

            idx_diff_low(I0) = idx_diff_low_const[I0] - do_borrow;

            idx_low(I0) += idx_diff_low[I0];
        }
        else
        {
            // not implemented
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
#if 1
        UpdateLowerIndex_1a(idx_diff_low, idx_diff_up, idx_low, idx_up_new, Number<Hack>{});
#elif 0
        UpdateLowerIndex_1b(idx_diff_low, idx_diff_up, idx_low, idx_up_new, Number<Hack>{});
#else
        UpdateLowerIndex_2(idx_diff_low, idx_diff_up, idx_low, idx_up_new, Number<Hack>{});
#endif
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsScan>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("DynamicMerge, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_scan_ ");
        print_multi_index(low_lengths_scan_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};
#else
template <typename LowLengths>
struct lambda_generate_magic_division_calculate_magic_multiplier
{
    template <index_t I>
    __host__ __device__ constexpr auto operator()(Number<I> i) const
    {
        return magic_division::CalculateMagicMultiplier(LowLengths{}[i]);
    }
};

template <typename LowLengths>
struct lambda_generate_magic_division_calculate_magic_shift
{
    template <index_t I>
    __host__ __device__ constexpr auto operator()(Number<I> i) const
    {
        return magic_division::CalculateMagicShift(LowLengths{}[i]);
    }
};

template <typename LowLengths>
struct DynamicMerge
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies_v2{}, Number<1>{})));

    using LowLengthsMagicDivisorMultipiler = decltype(
        generate_tuple(lambda_generate_magic_division_calculate_magic_multiplier<LowLengths>{},
                       Number<NDimLow>{}));

    using LowLengthsMagicDivisorShift = decltype(generate_tuple(
        lambda_generate_magic_division_calculate_magic_shift<LowLengths>{}, Number<NDimLow>{}));

    LowLengths low_lengths_;
    LowLengthsMagicDivisorMultipiler low_lengths_magic_divisor_multiplier_;
    LowLengthsMagicDivisorShift low_lengths_magic_divisor_shift_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr DynamicMerge() = default;

    __host__ __device__ constexpr DynamicMerge(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_magic_divisor_multiplier_{generate_tuple(
              [&](auto i) { return magic_division::CalculateMagicMultiplier(low_lengths[i]); },
              Number<NDimLow>{})},
          low_lengths_magic_divisor_shift_{generate_tuple(
              [&](auto i) { return magic_division::CalculateMagicShift(low_lengths[i]); },
              Number<NDimLow>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies_v2{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[Number<0>{}];

        static_for<NDimLow - 1, 0, -1>{}([&idx_low, &tmp, this](auto i) {
            index_t tmp2 =
                magic_division::DoMagicDivision(tmp,
                                                this->low_lengths_magic_divisor_multiplier_[i],
                                                this->low_lengths_magic_divisor_shift_[i]);
            idx_low(i) = tmp - tmp2 * this->low_lengths_[i];
            tmp        = tmp2;
        });

        idx_low(Number<0>{}) = tmp;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        auto idx_low_old = idx_low;

        CalculateLowerIndex(idx_low, idx_up_new);

        idx_diff_low = idx_low - idx_low_old;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsMagicDivisorMultipiler>::value &&
               is_known_at_compile_time<LowLengthsMagicDivisorShift>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("DynamicMerge, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_magic_divisor_multiplier_ ");
        print_multi_index(low_lengths_magic_divisor_multiplier_);
        printf("low_lengths_magic_divisor_shift_ ");
        print_multi_index(low_lengths_magic_divisor_shift_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};
#endif

template <typename UpLengths, bool Use24BitIntegerCalculation>
struct DynamicUnMerge
{
    static constexpr index_t NDimUp = UpLengths::Size();

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    using UpLengthsScan =
        decltype(container_reverse_exclusive_scan(UpLengths{}, math::multiplies_v2{}, Number<1>{}));

    UpLengths up_lengths_;
    UpLengthsScan up_lengths_scan_;

    __host__ __device__ constexpr DynamicUnMerge() = default;

    __host__ __device__ constexpr DynamicUnMerge(const UpLengths& up_lengths)
        : up_lengths_{up_lengths},
          up_lengths_scan_{
              container_reverse_exclusive_scan(up_lengths, math::multiplies_v2{}, Number<1>{})}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        if constexpr(!Use24BitIntegerCalculation)
        {
            idx_low(Number<0>{}) = idx_up[Number<NDimUp - 1>{}];

            static_for<0, NDimUp - 1, 1>{}(
                [&](auto i) { idx_low(Number<0>{}) += idx_up[i] * up_lengths_scan_[i]; });
        }
        else
        {
            idx_low(Number<0>{}) = idx_up[Number<NDimUp - 1>{}];

            static_for<0, NDimUp - 1, 1>{}([&](auto i) {
                idx_low(Number<0>{}) =
                    (0x00ffffff & idx_low[Number<0>{}]) +
                    (0x00ffffff & idx_up[i]) * (0x00ffffff & up_lengths_scan_[i]);
            });
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        CalculateLowerIndex(idx_diff_low, idx_diff_up);

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<UpLengthsScan>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("DynamicUnMerge, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        print_multi_index(up_lengths_scan_);
        printf("}");
    }
};

template <typename LowerIndex>
struct DynamicFreeze
{
    LowerIndex low_idx_;

    __host__ __device__ constexpr DynamicFreeze() = default;

    __host__ __device__ constexpr DynamicFreeze(const LowerIndex& low_idx) : low_idx_{low_idx} {}

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 0; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return Tuple<>{}; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low = low_idx_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx& idx_up_new,
                                                     Number<Hack>)
    {
        idx_diff_low(Number<0>{}) = index_t{Number<0>{}};
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowerIndex>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("DynamicFreeze");
        printf("low_idx_ %d", index_t{low_idx_});
    }
};

} // namespace ck
#endif
