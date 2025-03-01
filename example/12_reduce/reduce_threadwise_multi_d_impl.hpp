// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_threadwise_multi_d.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_reduce.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/host_common_util.hpp"

#include "reduce_example_common.hpp"

template <typename InOutDataType,
          typename AccDataType,
          ck::ReduceTensorOp ReduceOpId,
          ck::index_t Rank,
          ck::index_t NumReduceDim,
          bool PropagateNan,
          bool OutputIndex>
int reduce_threadwise_multi_d_impl(bool do_verification,
                                   int init_method,
                                   bool time_kernel,
                                   const std::vector<size_t>& inLengths,
                                   const std::array<int, NumReduceDim>& reduceDims,
                                   float alpha,
                                   float beta)

{
    using namespace ck;
    using namespace ck::tensor_operation::device;

    constexpr index_t NumOutDim = (Rank - NumReduceDim == 0) ? 1 : Rank - NumReduceDim;

    constexpr bool op_support_indices =
        (ReduceOpId == ReduceTensorOp::MIN || ReduceOpId == ReduceTensorOp::MAX ||
         ReduceOpId == ReduceTensorOp::AMAX);

    constexpr bool invalid_reduce_1 = OutputIndex && !op_support_indices;

    // 1) If InOutDataType is half_t, must use half_t as AccDataType for indexable reduction
    // operations 2) If InOutDataType is half_t, must use float as AccDataType for non-indexable
    // reduction operations
    constexpr bool invalid_reduce_2 =
        std::is_same<InOutDataType, half_t>::value &&
        ((!op_support_indices && !std::is_same<AccDataType, float>::value) ||
         (op_support_indices && !std::is_same<AccDataType, half_t>::value));

    // 1) If InOutDataType is float, must use float as AccDataType for indexable reduction
    // operations
    constexpr bool invalid_reduce_3 =
        std::is_same<InOutDataType, float>::value &&
        (op_support_indices && !std::is_same<AccDataType, float>::value);

    // 1) If InOutDataType is int8_t or int4_t, must use int8_t as AccDataType for indexable
    // reduction operations 2) If InOutDataType is int8_t or int4_t, must use int32_t as AccDataType
    // for non-indexable reduction operations
    constexpr bool invalid_reduce_4 =
        std::is_same<InOutDataType, int8_t>::value &&
        ((!op_support_indices && !std::is_same<AccDataType, int32_t>::value) ||
         (op_support_indices && !std::is_same<AccDataType, int8_t>::value));

    // 1) If InOutDataType is int8_t or int4_t, the supported operation must be either indexable
    // operations or ADD/AVG
    constexpr bool invalid_reduce_5 = std::is_same<InOutDataType, int8_t>::value &&
                                      (!op_support_indices && ReduceOpId != ReduceTensorOp::ADD &&
                                       ReduceOpId != ReduceTensorOp::AVG);

    // 1) If InOutDataType is bhalf_t, must use float as AccDataType for all reduction operations
    constexpr bool invalid_reduce_6 =
        std::is_same<InOutDataType, bhalf_t>::value && !std::is_same<AccDataType, float>::value;

    constexpr bool invalid_reduce = (invalid_reduce_1 || invalid_reduce_2 || invalid_reduce_3 ||
                                     invalid_reduce_4 || invalid_reduce_5 || invalid_reduce_6);

    if constexpr(invalid_reduce)
    {
        std::cerr << "The reduction setting is invalid, exiting!" << std::endl;
        return (-1);
    };

    using PassThrough = tensor_operation::element_wise::PassThrough;
    using Add         = tensor_operation::element_wise::Add;

    using ReduceOperation         = typename reduce_binary_operator<ReduceOpId>::opType;
    using InElementwiseOperation  = PassThrough;
    using OutElementwiseOperation = Add;

    using InOutDataTypeInDevice = InOutDataType;

    using DeviceReduceInstance =
        ck::tensor_operation::device::DeviceReduceThreadWiseMultiD<InOutDataTypeInDevice,
                                                                   ck::Tuple<InOutDataTypeInDevice>,
                                                                   AccDataType,
                                                                   InOutDataTypeInDevice,
                                                                   Rank,
                                                                   NumReduceDim,
                                                                   ReduceOperation,
                                                                   InElementwiseOperation,
                                                                   OutElementwiseOperation,
                                                                   256, // BlockSize
                                                                   4,   // MThreadSliceSize
                                                                   1,   // KThreadSliceSize
                                                                   0,   // InSrcVectorDim
                                                                   1,   // InSrceVectorSize
                                                                   1,
                                                                   Sequence<1>>; // OutDstVectorSize

    Tensor<InOutDataType> in(inLengths);

    std::vector<size_t> outLengths;

    auto invariantDims = get_invariant_dims<Rank, NumReduceDim>(reduceDims);

    if(invariantDims.empty())
        outLengths.push_back(1);
    else
        for(auto dim : invariantDims)
            outLengths.push_back(inLengths[dim]);

    Tensor<InOutDataType> out_ref(outLengths);
    Tensor<InOutDataType> out(outLengths);

    Tensor<InOutDataType> d0(outLengths);

    Tensor<int> out_indices_ref(outLengths);
    Tensor<int> out_indices(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = out.mDesc.GetStrides();

    size_t invariant_total_length = out.mDesc.GetElementSize();
    size_t reduce_total_length    = in.mDesc.GetElementSize() / invariant_total_length;

    std::size_t num_thread = 1;

    if(do_verification)
    {
        switch(init_method)
        {
        case 0: break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            d0.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            break;
        case 2:
            in.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            d0.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0, 5.0}, num_thread);
            d0.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0, 5.0}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0, 5.0},
                                            num_thread);
        }

        if(beta != 0.0f)
            for(size_t i = 0; i < out_ref.mDesc.GetElementSpaceSize(); i++)
                out.mData[i] = out_ref.mData[i];
    };

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(InOutDataTypeInDevice) * in.mDesc.GetElementSpaceSize());
    DeviceMem d0_dev(sizeof(InOutDataTypeInDevice) * d0.mDesc.GetElementSpaceSize());
    DeviceMem out_dev(sizeof(InOutDataTypeInDevice) * out.mDesc.GetElementSpaceSize());

    in_dev.ToDevice(in.mData.data());
    d0_dev.ToDevice(d0.mData.data());

    if(beta != 0.0f)
    {
        out_dev.ToDevice(out.mData.data());
    };

    size_t indicesSizeInBytes = OutputIndex ? out.mDesc.GetElementSize() * sizeof(int32_t) : 0;

    DeviceMem out_index_dev(indicesSizeInBytes);

    InElementwiseOperation in_elementwise_op;
    OutElementwiseOperation out_elementwise_op;

    std::array<index_t, Rank> arrInLengths;
    std::array<index_t, Rank> arrInStrides;

    std::array<index_t, NumOutDim> arrOutLengths;
    std::array<index_t, NumOutDim> arrOutStrides;

    ck::ranges::copy(inLengths, arrInLengths.begin());
    ck::ranges::copy(inStrides, arrInStrides.begin());

    ck::ranges::copy(outLengths, arrOutLengths.begin());
    ck::ranges::copy(outStrides, arrOutStrides.begin());

    if(do_verification)
    {
        using ReferenceReduceInstance =
            ck::tensor_operation::host::ReferenceReduce<InOutDataType,
                                                        AccDataType,
                                                        InOutDataType,
                                                        Rank,
                                                        NumReduceDim,
                                                        ReduceOperation,
                                                        InElementwiseOperation,
                                                        PassThrough,
                                                        PropagateNan,
                                                        OutputIndex>;

        auto reduce_ref = ReferenceReduceInstance{};

        auto argument_ptr_ref = reduce_ref.MakeArgumentPointer(arrInLengths,
                                                               arrInStrides,
                                                               arrOutLengths,
                                                               arrOutStrides,
                                                               reduceDims,
                                                               static_cast<double>(alpha),
                                                               static_cast<double>(beta),
                                                               in.mData.data(),
                                                               nullptr,
                                                               out_ref.mData.data(),
                                                               out_indices_ref.mData.data(),
                                                               in_elementwise_op,
                                                               PassThrough{});

        if(!reduce_ref.IsSupportedArgument(argument_ptr_ref.get()))
        {
            std::cout << "The runtime parameters not supported by the reduce reference, exiting!"
                      << std::endl;
            return (false);
        };

        auto invoker_ptr_ref = reduce_ref.MakeInvokerPointer();

        invoker_ptr_ref->Run(argument_ptr_ref.get());

        for(std::size_t i = 0; i < out_ref.GetElementSize(); i++)
            out_elementwise_op(out_ref.mData[i], out_ref.mData[i], d0.mData[i]);
    };

    auto reduce = DeviceReduceInstance{};

    auto argument_ptr = reduce.MakeArgumentPointer(arrInLengths,
                                                   arrInStrides,
                                                   {arrOutLengths},
                                                   {arrOutStrides},
                                                   arrOutLengths,
                                                   arrOutStrides,
                                                   reduceDims,
                                                   in_dev.GetDeviceBuffer(),
                                                   {d0_dev.GetDeviceBuffer()},
                                                   out_dev.GetDeviceBuffer(),
                                                   in_elementwise_op,
                                                   out_elementwise_op);

    if(!reduce.IsSupportedArgument(argument_ptr.get()))
    {
        std::cerr << "The runtime parameters not supported by the DeviceReduce instance, exiting!"
                  << std::endl;

        return (-2);
    };

    std::string reduce_name = reduce.GetTypeString();

    auto invoker_ptr = reduce.MakeInvokerPointer();

    float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    std::size_t num_bytes = invariant_total_length * reduce_total_length * sizeof(InOutDataType) +
                            invariant_total_length * sizeof(InOutDataType);

    float gb_per_sec = num_bytes / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << reduce_name
              << std::endl;

    bool pass = true;

    if(do_verification)
    {

        out_dev.FromDevice(out.mData.data());

        pass = pass && ck::utils::check_err(out, out_ref);

        if(OutputIndex)
        {
            out_index_dev.FromDevice(out_indices.mData.data());
            pass = pass && ck::utils::check_err(out_indices, out_indices_ref);
        };
    };

    return (pass ? 0 : 1);
}
