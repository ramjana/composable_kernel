#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_resize_concat_conv_bias_activ_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp"
#include "ck_conv_fig.h"

template <ck::index_t BlockSize_,
          ck::index_t E1_,
          ck::index_t E2_,
          ck::index_t K2_,
          ck::index_t E0PerBlock_,
          ck::index_t KPerBlock_,
          ck::index_t HoPerBlock_,
          ck::index_t WoPerBlock_,
          ck::index_t E1PerBlock_,
          ck::index_t KPerThread_,
          ck::index_t HoPerThread_,
          ck::index_t WoPerThread_,
          ck::index_t EPerThread_,
          typename ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2_,
          typename ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2_,
          ck::index_t ABlockTransferSrcScalarPerVector_E2_,
          ck::index_t ABlockTransferDstScalarPerVector_E2_,
          ck::index_t BThreadTransferSrcScalarPerVector_E2_,
          ck::index_t CThreadTransferDstScalarPerVector_K_>
struct GridGemmTuningParameters
{
    static constexpr auto BlockSize = BlockSize_;
    static constexpr auto E1        = E1_;
    static constexpr auto E2        = E2_;
    static constexpr auto K2        = K2_;

    static constexpr auto E0PerBlock = E0PerBlock_;
    static constexpr auto KPerBlock  = KPerBlock_;
    static constexpr auto HoPerBlock = HoPerBlock_;
    static constexpr auto WoPerBlock = WoPerBlock_;
    static constexpr auto E1PerBlock = E1PerBlock_;

    static constexpr auto KPerThread  = KPerThread_;
    static constexpr auto HoPerThread = HoPerThread_;
    static constexpr auto WoPerThread = WoPerThread_;
    static constexpr auto EPerThread  = EPerThread_;

    static constexpr auto ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2 =
        ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2_{};
    static constexpr auto ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 =
        ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2_{};

    static constexpr auto ABlockTransferSrcScalarPerVector_E2 =
        ABlockTransferSrcScalarPerVector_E2_;
    static constexpr auto ABlockTransferDstScalarPerVector_E2 =
        ABlockTransferDstScalarPerVector_E2_;
    static constexpr auto BThreadTransferSrcScalarPerVector_E2 =
        BThreadTransferSrcScalarPerVector_E2_;
    static constexpr auto CThreadTransferDstScalarPerVector_K =
        CThreadTransferDstScalarPerVector_K_;
};

template <typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
struct ConvDesc
{
    InLengths in_n_c0_hi_wi_c1_desc;
    WeiLengths wei_k_c0_y_x_c1_desc;
    OutLengths out_n_k0_ho_wo_k1_desc;

    ConvStrides conv_strides;
    ConvDilations conv_dilations;

    InLeftPads in_left_pads;
    InRightPads in_right_pads;

    ConvDesc(InLengths in_n_c0_hi_wi_c1_desc_,
             WeiLengths wei_k_c0_y_x_c1_desc_,
             OutLengths out_n_k0_ho_wo_k1_desc_,
             ConvStrides conv_strides_,
             ConvDilations conv_dilations_,
             InLeftPads in_left_pads_,
             InRightPads in_right_pads_)
    {
        in_n_c0_hi_wi_c1_desc  = in_n_c0_hi_wi_c1_desc_;
        wei_k_c0_y_x_c1_desc   = wei_k_c0_y_x_c1_desc_;
        out_n_k0_ho_wo_k1_desc = out_n_k0_ho_wo_k1_desc_;
        conv_strides           = conv_strides_;
        conv_dilations         = conv_dilations_;
        in_left_pads           = in_left_pads_;
        in_right_pads          = in_right_pads_;
    }
};

template <typename TInWei,
          typename TAcc,
          typename TOut,
          ck::ActivTypeEnum_t activ_type,
          typename In1Lengths,
          typename In2Lengths,
          typename Wei1Lengths,
          typename Wei2Lengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_resize_concat_conv_bias_activ_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1(
    const In1Lengths& in1_n_c0_hi_wi_c1_lengths,
    const In2Lengths& in2_n_c0_hi_wi_c1_lengths,
    const Wei1Lengths& wei1_k_c0_y_x_c1_lengths,
    const Wei2Lengths& wei2_k_c0_y_x_c1_lengths,
    const OutLengths& out_n_k0_ho_wo_k1_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in1_n_c0_hi_wi_c1,
    const Tensor<TInWei>& in2_n_c0_hi_wi_c1,
    const Tensor<TInWei>& wei1_k_c0_y_x_c1,
    const Tensor<TInWei>& wei2_k_c0_y_x_c1,
    const Tensor<TOut>& bias_k0_k1,
    Tensor<TOut>& out_n_k0_ho_wo_k1,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    const auto N  = out_n_k0_ho_wo_k1_lengths[I0];
    const auto K0 = out_n_k0_ho_wo_k1_lengths[I1];
    const auto Ho = out_n_k0_ho_wo_k1_lengths[I2];
    const auto Wo = out_n_k0_ho_wo_k1_lengths[I3];
    const auto K1 = out_n_k0_ho_wo_k1_lengths[I4];

    const auto CONV1_C0 = in1_n_c0_hi_wi_c1_lengths[I1];
    const auto Hi       = in1_n_c0_hi_wi_c1_lengths[I2];
    const auto Wi       = in1_n_c0_hi_wi_c1_lengths[I3];
    const auto C1       = in1_n_c0_hi_wi_c1_lengths[I4];

    const auto CONV2_C0 = in2_n_c0_hi_wi_c1_lengths[I1];
    // const auto CONV2_Hi = in2_n_c0_hi_wi_c1_lengths[I2];
    // const auto CONV2_Wi = in2_n_c0_hi_wi_c1_lengths[I3];
    // const auto CONV2_C1 = in2_n_c0_hi_wi_c1_lengths[I4];

    const auto K = wei1_k_c0_y_x_c1_lengths[I0];
    const auto Y = wei1_k_c0_y_x_c1_lengths[I2];
    const auto X = wei1_k_c0_y_x_c1_lengths[I3];

    DeviceMem in1_n_c0_hi_wi_c1_device_buf(sizeof(TInWei) *
                                           in1_n_c0_hi_wi_c1.mDesc.GetElementSpace());
    DeviceMem in2_n_c0_hi_wi_c1_device_buf(sizeof(TInWei) *
                                           in2_n_c0_hi_wi_c1.mDesc.GetElementSpace());
    DeviceMem wei1_k_c0_y_x_c1_device_buf(sizeof(TInWei) *
                                          wei1_k_c0_y_x_c1.mDesc.GetElementSpace());
    DeviceMem wei2_k_c0_y_x_c1_device_buf(sizeof(TInWei) *
                                          wei2_k_c0_y_x_c1.mDesc.GetElementSpace());
    DeviceMem bias_k0_k1_device_buf(sizeof(TOut) * bias_k0_k1.mDesc.GetElementSpace());
    DeviceMem out_n_k0_ho_wo_k1_device_buf(sizeof(TOut) *
                                           out_n_k0_ho_wo_k1.mDesc.GetElementSpace());
    in1_n_c0_hi_wi_c1_device_buf.ToDevice(in1_n_c0_hi_wi_c1.mData.data());
    in2_n_c0_hi_wi_c1_device_buf.ToDevice(in2_n_c0_hi_wi_c1.mData.data());
    wei1_k_c0_y_x_c1_device_buf.ToDevice(wei1_k_c0_y_x_c1.mData.data());
    wei2_k_c0_y_x_c1_device_buf.ToDevice(wei2_k_c0_y_x_c1.mData.data());
    bias_k0_k1_device_buf.ToDevice(bias_k0_k1.mData.data());

    // blocksize = 256
#if 0 
    constexpr index_t BlockSize = CONV_BLOCK_SIZE;

    constexpr index_t E1 = CONV_E1;
    constexpr index_t E2 = CONV_E2;
    constexpr index_t K2 = CONV_K2;

    constexpr index_t E0PerBlock = CONV_E0_PER_BLOCK;
    constexpr index_t KPerBlock  = CONV_K_PER_BLOCK;
    constexpr index_t HoPerBlock = CONV_HO_PER_BLOCK;
    constexpr index_t WoPerBlock = CONV_WO_PER_BLOCK;
    constexpr index_t E1PerBlock = CONV_E1_PER_BLOCK;

    constexpr index_t KPerThread  = CONV_K_PER_THREAD;
    constexpr index_t HoPerThread = CONV_HO_PER_THREAD;
    constexpr index_t WoPerThread = CONV_WO_PER_THREAD;
    constexpr index_t EPerThread  = CONV_E_PER_THREAD;

    using ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2 =
        Sequence<CONV_ABLOCK_TRANS_THREAD_SLICE_LENGTHS>;
    using ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 =
        Sequence<CONV_ABLOCK_TRANS_THREAD_CLUSTER_LENGTHS>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E2  = C1;
    constexpr index_t ABlockTransferDstScalarPerVector_E2  = C1;
    constexpr index_t BThreadTransferSrcScalarPerVector_E2 = C1;
    constexpr index_t CThreadTransferDstScalarPerVector_K  = K1;
#endif

    GridGemmTuningParameters<
        256,                             // BlockSize
        CONV1_C0 * Y * X,                // E1
        C1,                              // E2
        4,                               // K2
        1,                               // E0PerBlock
        16,                              // KPerBlock
        16,                              // HoPerBlock
        64,                              // WoPerBlock
        2,                               // E1PerBlock
        16,                              // KPerThread
        2,                               // HoPerThread
        2,                               // WoPerThread
        1,                               // EPerThread
        Sequence<1, Y * X, 1, 1, C1>,    // ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2
        Sequence<1, CONV1_C0, 1, 16, 1>, // ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2
        C1,                              // ABlockTransferSrcScalarPerVector_E2
        C1,                              // ABlockTransferDstScalarPerVector_E2
        C1,                              // BThreadTransferSrcScalarPerVector_E2
        C1                               // CThreadTransferDstScalarPerVector_K
        >
        conv1_tuning_parameters{};

    GridGemmTuningParameters<
        256,                             // BlockSize
        CONV2_C0 * Y * X,                // E1
        C1,                              // E2
        4,                               // K2
        1,                               // E0PerBlock
        16,                              // KPerBlock
        16,                              // HoPerBlock
        64,                              // WoPerBlock
        2,                               // E1PerBlock
        16,                              // KPerThread
        2,                               // HoPerThread
        2,                               // WoPerThread
        1,                               // EPerThread
        Sequence<1, Y * X, 1, 1, C1>,    // ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2
        Sequence<1, CONV2_C0, 1, 16, 1>, // ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2
        C1,                              // ABlockTransferSrcScalarPerVector_E2
        C1,                              // ABlockTransferDstScalarPerVector_E2
        C1,                              // BThreadTransferSrcScalarPerVector_E2
        C1                               // CThreadTransferDstScalarPerVector_K
        >
        conv2_tuning_parameters{};

    const auto in1_n_c0_hi_wi_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, CONV1_C0, Hi, Wi, C1));
    const auto wei1_k_c0_y_x_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(K, CONV1_C0, Y, X, C1));
    const auto out_n_k0_ho_wo_k1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1));

    static_assert(in1_n_c0_hi_wi_c1_desc.IsKnownAtCompileTime(), "");
    static_assert(wei1_k_c0_y_x_c1_desc.IsKnownAtCompileTime(), "");
    static_assert(out_n_k0_ho_wo_k1_desc.IsKnownAtCompileTime(), "");

    ConvDesc conv1_desc(in1_n_c0_hi_wi_c1_desc,
                        wei1_k_c0_y_x_c1_desc,
                        out_n_k0_ho_wo_k1_desc,
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads);

    const auto in2_n_c0_hi_wi_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, CONV2_C0, Hi, Wi, C1));
    const auto wei2_k_c0_y_x_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(K, CONV2_C0, Y, X, C1));

    static_assert(in2_n_c0_hi_wi_c1_desc.IsKnownAtCompileTime(), "");
    static_assert(wei2_k_c0_y_x_c1_desc.IsKnownAtCompileTime(), "");

    ConvDesc conv2_desc(in2_n_c0_hi_wi_c1_desc,
                        wei2_k_c0_y_x_c1_desc,
                        out_n_k0_ho_wo_k1_desc,
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads);
#if 1
    constexpr auto conv_driver =
        DriverDynamicResizeConcatConvBiasActivForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1<
            TInWei,
            TAcc,
            TOut,
            decltype(conv1_tuning_parameters),
            decltype(conv2_tuning_parameters),
            I1,
            activ_type>{};

    for(int i = 0; i < 5; i++)
    {

        // const auto ave_time =
        // conv_driver.Run(conv1_desc,
        // conv2_desc,
        // static_cast<TInWei*>(wei1_k_c0_y_x_c1_device_buf.GetDeviceBuffer()),
        // static_cast<TInWei*>(in1_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
        // static_cast<TInWei*>(wei2_k_c0_y_x_c1_device_buf.GetDeviceBuffer()),
        // static_cast<TInWei*>(in2_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
        // static_cast<TOut*>(bias_k0_k1_device_buf.GetDeviceBuffer()),
        // static_cast<TOut*>(out_n_k0_ho_wo_k1_device_buf.GetDeviceBuffer()),
        // nrepeat);

        const auto ave_time = conv_driver.Run_test(
            conv1_desc,
            static_cast<TInWei*>(wei1_k_c0_y_x_c1_device_buf.GetDeviceBuffer()),
            static_cast<TInWei*>(in1_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
            static_cast<TOut*>(bias_k0_k1_device_buf.GetDeviceBuffer()),
            static_cast<TOut*>(out_n_k0_ho_wo_k1_device_buf.GetDeviceBuffer()),
            nrepeat);

        {
            // float perf = static_cast<float>(std::size_t(2) * N * K * Ho * Wo * C0 * C1 * Y * X) /
            //(std::size_t(1000) * 1000 * 1000) / ave_time;

            // std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
            //<< std::endl;
        }
    }
#endif

    out_n_k0_ho_wo_k1_device_buf.FromDevice(out_n_k0_ho_wo_k1.mData.data());
}
