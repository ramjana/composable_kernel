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

    void printTuningParameters()
    {
        using namespace ck;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        std::cout << "BlockSize_" << BlockSize << "_E1_" << E1 << "_E2_" << E2 << "_K2_" << K2
                  << "_KPerBlock_" << KPerBlock << "_HoPerBlock_" << HoPerBlock << "_WoPerBlock_"
                  << WoPerBlock << "_E0PerBlock_" << E0PerBlock << "_E1PerBlock_" << E1PerBlock
                  << "_KPerThread_" << KPerThread << "_HoPerThread_" << HoPerThread
                  << "_WoPerThread_" << WoPerThread << "_EPerThread_" << EPerThread
                  << "_ABlockTransferThreadSliceLengths_<"
                  << ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I0] << "_"
                  << ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I1] << "_"
                  << ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I2] << "_"
                  << ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I3] << "_"
                  << ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I4] << ">"
                  << "_ABlockTransferThreadClusterLengths_<"
                  << ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I0] << "_"
                  << ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I1] << "_"
                  << ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I2] << "_"
                  << ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I3] << "_"
                  << ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I4] << ">"
                  << "_ABlockTransferSrcScalarPerVector_E2_" << ABlockTransferSrcScalarPerVector_E2
                  << "_ABlockTransferDstScalarPerVector_E2_" << ABlockTransferDstScalarPerVector_E2
                  << "_BThreadTransferSrcScalarPerVector_E2_"
                  << BThreadTransferSrcScalarPerVector_E2 << "_CThreadTransferDstScalarPerVector_K_"
                  << CThreadTransferDstScalarPerVector_K << std::endl;
    }
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
    WeiLengths wei_c0_y_x_k_c1_desc;
    OutLengths out_n_k0_ho_wo_k1_desc;

    ConvStrides conv_strides;
    ConvDilations conv_dilations;

    InLeftPads in_left_pads;
    InRightPads in_right_pads;

    ConvDesc(InLengths in_n_c0_hi_wi_c1_desc_,
             WeiLengths wei_c0_y_x_k_c1_desc_,
             OutLengths out_n_k0_ho_wo_k1_desc_,
             ConvStrides conv_strides_,
             ConvDilations conv_dilations_,
             InLeftPads in_left_pads_,
             InRightPads in_right_pads_)
    {
        in_n_c0_hi_wi_c1_desc  = in_n_c0_hi_wi_c1_desc_;
        wei_c0_y_x_k_c1_desc   = wei_c0_y_x_k_c1_desc_;
        out_n_k0_ho_wo_k1_desc = out_n_k0_ho_wo_k1_desc_;
        conv_strides           = conv_strides_;
        conv_dilations         = conv_dilations_;
        in_left_pads           = in_left_pads_;
        in_right_pads          = in_right_pads_;
    }

    void printConvDesc()
    {
        using namespace ck;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        const auto N  = in_n_c0_hi_wi_c1_desc.GetLength(I0);
        const auto C0 = in_n_c0_hi_wi_c1_desc.GetLength(I1);
        const auto Hi = in_n_c0_hi_wi_c1_desc.GetLength(I2);
        const auto Wi = in_n_c0_hi_wi_c1_desc.GetLength(I3);
        const auto C1 = in_n_c0_hi_wi_c1_desc.GetLength(I4);

        const auto K0 = out_n_k0_ho_wo_k1_desc.GetLength(I1);
        const auto Ho = out_n_k0_ho_wo_k1_desc.GetLength(I2);
        const auto Wo = out_n_k0_ho_wo_k1_desc.GetLength(I3);
        const auto K1 = out_n_k0_ho_wo_k1_desc.GetLength(I4);

        const auto K = wei_c0_y_x_k_c1_desc.GetLength(I0);
        const auto Y = wei_c0_y_x_k_c1_desc.GetLength(I2);
        const auto X = wei_c0_y_x_k_c1_desc.GetLength(I3);

        const auto ConvStrideH = conv_strides[I0];
        const auto ConvStrideW = conv_strides[I1];

        const auto ConvDilationH = conv_dilations[I0];
        const auto ConvDilationW = conv_dilations[I1];

        std::cout << "input_"
                  << "n" << N << "c" << C0 << "h" << Hi << "w" << Wi << "c" << C1 << "_filter_k"
                  << K << "c" << C0 << "y" << Y << "x" << X << "c" << C1 << "_out_n" << N << "k"
                  << K0 << "h" << Ho << "w" << Wo << "k" << K1 << std::endl;

        std::cout << "ConvStride = " << ConvStrideH << "," << ConvStrideW << std::endl;
        std::cout << "ConvDilation = " << ConvDilationH << "," << ConvDilationW << std::endl;
    }
};

template <typename TInWei,
          typename TAcc,
          typename TOut,
          ck::ActivTypeEnum_t activ_type,
          typename In1Lengths,
          typename In2Lengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_convolution_bias_activ_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1(
    const In1Lengths& in1_n_c0_hi_wi_c1_lengths,
    const In2Lengths& in2_n_c0_hi_wi_c1_lengths,
    const WeiLengths& wei_c0_y_x_k_c1_lengths,
    const OutLengths& out_n_k0_ho_wo_k1_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in1_n_c0_hi_wi_c1,
    const Tensor<TInWei>& in2_n_c0_hi_wi_c1,
    const Tensor<TInWei>& wei_c0_y_x_k_c1,
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
    const auto CONV2_C0 = in2_n_c0_hi_wi_c1_lengths[I1];

    const auto Hi = in1_n_c0_hi_wi_c1_lengths[I2];
    const auto Wi = in1_n_c0_hi_wi_c1_lengths[I3];
    const auto C1 = in1_n_c0_hi_wi_c1_lengths[I4];

    const auto C0 = wei_c0_y_x_k_c1_lengths[I0];
    const auto Y  = wei_c0_y_x_k_c1_lengths[I1];
    const auto X  = wei_c0_y_x_k_c1_lengths[I2];
    const auto K  = wei_c0_y_x_k_c1_lengths[I3];

    DeviceMem in1_n_c0_hi_wi_c1_device_buf(sizeof(TInWei) *
                                           in1_n_c0_hi_wi_c1.mDesc.GetElementSpace());
    DeviceMem bias_k0_k1_device_buf(sizeof(TOut) * bias_k0_k1.mDesc.GetElementSpace());
    DeviceMem out_n_k0_ho_wo_k1_device_buf(sizeof(TOut) *
                                           out_n_k0_ho_wo_k1.mDesc.GetElementSpace());

    in1_n_c0_hi_wi_c1_device_buf.ToDevice(in1_n_c0_hi_wi_c1.mData.data());

    bias_k0_k1_device_buf.ToDevice(bias_k0_k1.mData.data());

    const auto in1_n_c0_hi_wi_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, CONV1_C0, Hi, Wi, C1));

    const auto out_n_k0_ho_wo_k1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1));

    DeviceMem in2_n_c0_hi_wi_c1_device_buf(sizeof(TInWei) *
                                           in2_n_c0_hi_wi_c1.mDesc.GetElementSpace());

    in2_n_c0_hi_wi_c1_device_buf.ToDevice(in2_n_c0_hi_wi_c1.mData.data());

    const auto in2_n_c0_hi_wi_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, CONV2_C0, Hi, Wi, C1));

    DeviceMem wei_c0_y_x_k_c1_device_buf(sizeof(TInWei) * wei_c0_y_x_k_c1.mDesc.GetElementSpace());
    wei_c0_y_x_k_c1_device_buf.ToDevice(wei_c0_y_x_k_c1.mData.data());

    const auto wei1_c0_y_x_k_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(CONV1_C0, Y, X, K, C1));
    const auto wei2_c0_y_x_k_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(CONV2_C0, Y, X, K, C1));

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
        Sequence<1, CONV1_C0, 1, 1, C1>, // ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2
        Sequence<1, Y * X, 1, 16, 1>,    // ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2
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
        Sequence<1, CONV2_C0, 1, 1, C1>, // ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2
        Sequence<1, Y * X, 1, 16, 1>,    // ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2
        C1,                              // ABlockTransferSrcScalarPerVector_E2
        C1,                              // ABlockTransferDstScalarPerVector_E2
        C1,                              // BThreadTransferSrcScalarPerVector_E2
        C1                               // CThreadTransferDstScalarPerVector_K
        >
        conv2_tuning_parameters{};

    constexpr auto conv_driver =
        DriverDynamicResizeConcatConvBiasActivForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1<
            TInWei,
            TAcc,
            TOut,
            decltype(conv1_tuning_parameters),
            decltype(conv2_tuning_parameters),
            I1,
            activ_type>{};

    ConvDesc conv1_desc(in1_n_c0_hi_wi_c1_desc,
                        wei1_c0_y_x_k_c1_desc,
                        out_n_k0_ho_wo_k1_desc,
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads);

    ConvDesc conv2_desc(in2_n_c0_hi_wi_c1_desc,
                        wei2_c0_y_x_k_c1_desc,
                        out_n_k0_ho_wo_k1_desc,
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads);

    conv1_tuning_parameters.printTuningParameters();
    conv2_tuning_parameters.printTuningParameters();

    conv1_desc.printConvDesc();
    conv2_desc.printConvDesc();

    for(int i = 0; i < 5; i++)
    {
        const auto ave_time =
            conv_driver.Run(conv1_desc,
                            conv2_desc,
                            static_cast<TInWei*>(wei_c0_y_x_k_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TInWei*>(in1_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TInWei*>(in2_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(bias_k0_k1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(out_n_k0_ho_wo_k1_device_buf.GetDeviceBuffer()),
                            nrepeat);
        {
            float perf = static_cast<float>(std::size_t(2) * N * K * Ho * Wo * C0 * C1 * Y * X) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }

    out_n_k0_ho_wo_k1_device_buf.FromDevice(out_n_k0_ho_wo_k1.mData.data());
}
