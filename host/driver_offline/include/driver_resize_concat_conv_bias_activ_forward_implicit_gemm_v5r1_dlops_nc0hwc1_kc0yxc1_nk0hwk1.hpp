#ifndef DRIVER_RESIZE_CONCAT_CONV_BIAS_ACTIV_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NC0HWc1_KC0YXC1_NK0HWK1_HPP
#define DRIVER_RESIZE_CONCAT_CONV_BIAS_ACTIV_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NC0HWc1_KC0YXC1_NK0HWK1_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_dlops_v3.hpp"

namespace ck {

template <typename GridGemmTuningParameters_>
void printTuningParameters()
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    std::cout
        << "BlockSize_" << GridGemmTuningParameters_::BlockSize << "_E1_"
        << GridGemmTuningParameters_::E1 << "_E2_" << GridGemmTuningParameters_::E2 << "_K2_"
        << GridGemmTuningParameters_::K2 << "_KPerBlock_" << GridGemmTuningParameters_::KPerBlock
        << "_HoPerBlock_" << GridGemmTuningParameters_::HoPerBlock << "_WoPerBlock_"
        << GridGemmTuningParameters_::WoPerBlock << "_E0PerBlock_"
        << GridGemmTuningParameters_::E0PerBlock << "_E1PerBlock_"
        << GridGemmTuningParameters_::E1PerBlock << "_KPerThread_"
        << GridGemmTuningParameters_::KPerThread << "_HoPerThread_"
        << GridGemmTuningParameters_::HoPerThread << "_WoPerThread_"
        << GridGemmTuningParameters_::WoPerThread << "_EPerThread_"
        << GridGemmTuningParameters_::EPerThread << "_ABlockTransferThreadSliceLengths_<"
        << GridGemmTuningParameters_::ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I0] << "_"
        << GridGemmTuningParameters_::ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I1] << "_"
        << GridGemmTuningParameters_::ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I2] << "_"
        << GridGemmTuningParameters_::ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I3] << "_"
        << GridGemmTuningParameters_::ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I4] << ">"
        << "_ABlockTransferThreadClusterLengths_<"
        << GridGemmTuningParameters_::ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I0] << "_"
        << GridGemmTuningParameters_::ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I1] << "_"
        << GridGemmTuningParameters_::ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I2] << "_"
        << GridGemmTuningParameters_::ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I3] << "_"
        << GridGemmTuningParameters_::ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I4] << ">"
        << std::endl;
}

template <typename GemmDesc_>
void printGemmDesc(GemmDesc_ conv_desc)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    const auto in_n_c0_hi_wi_c1_global_desc = conv_desc.in_n_c0_hi_wi_c1_desc;

    const auto N  = in_n_c0_hi_wi_c1_global_desc.GetLength(I0);
    const auto C0 = in_n_c0_hi_wi_c1_global_desc.GetLength(I1);
    const auto Hi = in_n_c0_hi_wi_c1_global_desc.GetLength(I2);
    const auto Wi = in_n_c0_hi_wi_c1_global_desc.GetLength(I3);
    const auto C1 = in_n_c0_hi_wi_c1_global_desc.GetLength(I4);

    const auto out_n_k0_ho_wo_k1_global_desc = conv_desc.out_n_k0_ho_wo_k1_desc;

    const auto K0 = out_n_k0_ho_wo_k1_global_desc.GetLength(I1);
    const auto Ho = out_n_k0_ho_wo_k1_global_desc.GetLength(I2);
    const auto Wo = out_n_k0_ho_wo_k1_global_desc.GetLength(I3);
    const auto K1 = out_n_k0_ho_wo_k1_global_desc.GetLength(I4);

    const auto wei_k_c0_y_x_c1_global_desc = conv_desc.out_n_k0_ho_wo_k1_desc;

    const auto K = wei_k_c0_y_x_c1_global_desc.GetLength(I0);
    const auto Y = wei_k_c0_y_x_c1_global_desc.GetLength(I2);
    const auto X = wei_k_c0_y_x_c1_global_desc.GetLength(I3);

    const auto ConvStrideH = conv_desc.conv_strides[I0];
    const auto ConvStrideW = conv_desc.conv_strides[I1];

    const auto ConvDilationH = conv_desc.conv_dilations[I0];
    const auto ConvDilationW = conv_desc.conv_dilations[I1];

    std::cout << "input_"
              << "n" << N << "c" << C0 << "h" << Hi << "w" << Wi << "c" << C1 << "_filter_k" << K
              << "c" << C0 << "y" << Y << "x" << X << "c" << C1 << "_out_n" << N << "k" << K0 << "h"
              << Ho << "w" << Wo << "k" << K1 << std::endl;

    std::cout << "ConvStride = " << ConvStrideH << "," << ConvStrideW << std::endl;
    std::cout << "ConvDilation = " << ConvDilationH << "," << ConvDilationW << std::endl;
}

template <typename GridwiseGemmDesc,
          typename AE0E1K0K1E2GridDesc,
          typename BE0E1NH0H1H2W0W1W2E2GridDesc,
          typename CK0K1NH0H1H2W0W1W2GridDesc,
          typename CBlockIdToKNHoWoBlockClusterAdaptor>
struct GemmArguments
{
    // static constexpr auto gemmwise_gemm_desc = GridwiseGemmDesc{};

    // GridwiseGemmDesc gridwise_gemm_desc;
    // AE0E1K0K1E2GridDesc a_e0_e1_k0_k1_e2_grid_desc;
    // BE0E1NH0H1H2W0W1W2E2GridDesc b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc;
    // CK0K1NH0H1H2W0W1W2GridDesc c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc;
    // CBlockIdToKNHoWoBlockClusterAdaptor c_blockid_to_k_n_h_w_block_cluster_adaptor;

    static constexpr auto gridwise_gemm_desc                       = GridwiseGemmDesc{};
    static constexpr auto a_e0_e1_k0_k1_e2_grid_desc               = AE0E1K0K1E2GridDesc{};
    static constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc = BE0E1NH0H1H2W0W1W2E2GridDesc{};
    static constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToKNHoWoBlockClusterAdaptor{};

    index_t grid_size;
    bool has_main_e0_block_loop;

    GemmArguments(
        // const GridwiseGemmDesc gridwise_gemm_desc_,
        // const AE0E1K0K1E2GridDesc a_e0_e1_k0_k1_e2_grid_desc_,
        // const BE0E1NH0H1H2W0W1W2E2GridDesc b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc_,
        // const CK0K1NH0H1H2W0W1W2GridDesc c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc_,
        // const CBlockIdToKNHoWoBlockClusterAdaptor c_blockid_to_k_n_h_w_block_cluster_adaptor_,
        const index_t grid_size_,
        const bool has_main_e0_block_loop_)
    {
        // gridwise_gemm_desc = gridwise_gemm_desc_;

        // a_e0_e1_k0_k1_e2_grid_desc                 = a_e0_e1_k0_k1_e2_grid_desc_;
        // b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc   = b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc_;
        // c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc      = c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc_;
        // c_blockid_to_k_n_h_w_block_cluster_adaptor = c_blockid_to_k_n_h_w_block_cluster_adaptor_;

        grid_size              = grid_size_;
        has_main_e0_block_loop = has_main_e0_block_loop_;
    }
};

template <typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          typename GridGemmTuningParameters,
          typename ConvDesc>
constexpr auto MakeGridwiseGemm(ConvDesc conv_desc)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    printTuningParameters<GridGemmTuningParameters>();
    printGemmDesc(conv_desc);

    constexpr index_t BlockSize = GridGemmTuningParameters::BlockSize;

    constexpr index_t E1 = GridGemmTuningParameters::E1;
    constexpr index_t E2 = GridGemmTuningParameters::E2;
    constexpr index_t K2 = GridGemmTuningParameters::K2;

    constexpr index_t E0PerBlock = GridGemmTuningParameters::E0PerBlock;
    constexpr index_t KPerBlock  = GridGemmTuningParameters::KPerBlock;
    constexpr index_t HoPerBlock = GridGemmTuningParameters::HoPerBlock;
    constexpr index_t WoPerBlock = GridGemmTuningParameters::WoPerBlock;
    constexpr index_t E1PerBlock = GridGemmTuningParameters::E1PerBlock;

    constexpr index_t KPerThread  = GridGemmTuningParameters::KPerThread;
    constexpr index_t HoPerThread = GridGemmTuningParameters::HoPerThread;
    constexpr index_t WoPerThread = GridGemmTuningParameters::WoPerThread;
    constexpr index_t EPerThread  = GridGemmTuningParameters::EPerThread;

    using ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2 =
        decltype(GridGemmTuningParameters::ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2);
    using ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 =
        decltype(GridGemmTuningParameters::ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2);

    constexpr auto ABlockTransferSrcScalarPerVector_E2 =
        GridGemmTuningParameters::ABlockTransferSrcScalarPerVector_E2;
    constexpr auto ABlockTransferDstScalarPerVector_E2 =
        GridGemmTuningParameters::ABlockTransferDstScalarPerVector_E2;
    constexpr auto BThreadTransferSrcScalarPerVector_E2 =
        GridGemmTuningParameters::BThreadTransferSrcScalarPerVector_E2;
    constexpr auto CThreadTransferDstScalarPerVector_K =
        GridGemmTuningParameters::CThreadTransferDstScalarPerVector_K;

    constexpr auto in_n_c0_hi_wi_c1_global_desc = conv_desc.in_n_c0_hi_wi_c1_desc;

    constexpr index_t N  = in_n_c0_hi_wi_c1_global_desc.GetLength(I0);
    constexpr index_t C0 = in_n_c0_hi_wi_c1_global_desc.GetLength(I1);
    constexpr index_t Hi = in_n_c0_hi_wi_c1_global_desc.GetLength(I2);
    constexpr index_t Wi = in_n_c0_hi_wi_c1_global_desc.GetLength(I3);
    constexpr index_t C1 = in_n_c0_hi_wi_c1_global_desc.GetLength(I4);

    constexpr auto out_n_k0_ho_wo_k1_global_desc = conv_desc.out_n_k0_ho_wo_k1_desc;

    constexpr index_t K0 = out_n_k0_ho_wo_k1_global_desc.GetLength(I1);
    constexpr index_t Ho = out_n_k0_ho_wo_k1_global_desc.GetLength(I2);
    constexpr index_t Wo = out_n_k0_ho_wo_k1_global_desc.GetLength(I3);
    constexpr index_t K1 = out_n_k0_ho_wo_k1_global_desc.GetLength(I4);

    constexpr auto wei_k_c0_y_x_c1_global_desc = conv_desc.wei_k_c0_y_x_c1_desc;

    constexpr index_t K = wei_k_c0_y_x_c1_global_desc.GetLength(I0);
    constexpr index_t Y = wei_k_c0_y_x_c1_global_desc.GetLength(I2);
    constexpr index_t X = wei_k_c0_y_x_c1_global_desc.GetLength(I3);

    constexpr index_t ConvStrideH = conv_desc.conv_strides[I0];
    constexpr index_t ConvStrideW = conv_desc.conv_strides[I1];

    constexpr index_t ConvDilationH = conv_desc.conv_dilations[I0];
    constexpr index_t ConvDilationW = conv_desc.conv_dilations[I1];

#if CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR
    constexpr index_t Hop = Number<(Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock>{};
    constexpr index_t Wop = Number<(Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock>{};
#else
    constexpr index_t Hop = (Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock;
    constexpr index_t Wop = (Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock;
#endif

    constexpr index_t InLeftPadH = conv_desc.in_left_pads[I0];
    constexpr index_t InLeftPadW = conv_desc.in_left_pads[I1];

    constexpr index_t OutRightPadH = Hop - Ho;
    constexpr index_t OutRightPadW = Wop - Wo;

    constexpr index_t InRightPadH = conv_desc.in_right_pads[I0] + OutRightPadH * ConvStrideH;
    constexpr index_t InRightPadW = conv_desc.in_right_pads[I1] + OutRightPadW * ConvStrideW;

    if((C0 * Y * X) % (E1 * E0PerBlock) != 0)
    {
        throw std::runtime_error("wrong! GEMM size no divisible");
    }

    constexpr index_t E  = C0 * Y * X;
    constexpr index_t E0 = E / E1;

    // weight tensor
    constexpr auto a_e_k_e2_grid_desc =
        transform_tensor_descriptor(make_naive_tensor_descriptor_packed(make_tuple(K, E, E2)),
                                    make_tuple(make_pass_through_transform(K),
                                               make_pass_through_transform(C0 * Y * X),
                                               make_pass_through_transform(E2)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                    make_tuple(Sequence<1>{}, Sequence<0>{}, Sequence<2>{}));

    constexpr auto a_e0_e1_k_e2_grid_desc =
        transform_tensor_descriptor(a_e_k_e2_grid_desc,
                                    make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                                               make_pass_through_transform(K),
                                               make_pass_through_transform(E2)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

    // input tensor
    constexpr auto in_n_c0_hip_wip_e2_global_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi, E2)),
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C0),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW),
                   make_pass_through_transform(E2)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

    constexpr auto in_n_c0_y_ho_x_wo_e2_global_desc = transform_tensor_descriptor(
        in_n_c0_hip_wip_e2_global_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C0),
                   make_embed_transform(make_tuple(Y, Hop), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wop), make_tuple(ConvDilationW, ConvStrideW)),
                   make_pass_through_transform(E2)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(
            Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}, Sequence<6>{}));

    constexpr auto in_e_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
        in_n_c0_y_ho_x_wo_e2_global_desc,
        make_tuple(make_merge_transform(make_tuple(C0, Y, X)),
                   make_pass_through_transform(N),
                   make_pass_through_transform(Hop),
                   make_pass_through_transform(Wop),
                   make_pass_through_transform(E2)),
        make_tuple(Sequence<1, 2, 4>{}, Sequence<0>{}, Sequence<3>{}, Sequence<5>{}, Sequence<6>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

    constexpr auto b_e0_e1_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
        in_e_n_ho_wo_e2_grid_desc,
        make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                   make_pass_through_transform(N),
                   make_pass_through_transform(Hop),
                   make_pass_through_transform(Wop),
                   make_pass_through_transform(E2)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}));

    // output tensor
    constexpr auto c_k_n_hop_wop_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1)),
        make_tuple(make_merge_transform(make_tuple(K0, K1)),
                   make_pass_through_transform(N),
                   make_pad_transform(Ho, I0, OutRightPadH),
                   make_pad_transform(Wo, I0, OutRightPadW)),
        make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    std::cerr << "Hop = " << Hop << " Wop = " << Wop << std::endl;

    if(!((K % KPerBlock) == 0 && (Hop % HoPerBlock) == 0 && (Wop % WoPerBlock) == 0 &&
         (E1 % E1PerBlock) == 0))
    {
        throw std::runtime_error("wrong! GEMM size no divisible");
    }

    std::cerr << "a_size = " << a_e0_e1_k_e2_grid_desc.GetElementSpaceSize() * sizeof(FloatAB)
              << ", b_size = "
              << b_e0_e1_n_ho_wo_e2_grid_desc.GetElementSpaceSize() * sizeof(FloatAB)
              << ", c = " << c_k_n_hop_wop_grid_desc.GetElementSpaceSize() * sizeof(FloatC)
              << std::endl;

    // GEMM
    using GridwiseGemm = GridwiseGemmDlops_km_kn_mn_v3<
        BlockSize,
        FloatAB,
        FloatAcc,
        FloatC,
        InMemoryDataOperationEnum_t::Set,
        decltype(a_e0_e1_k_e2_grid_desc),
        decltype(b_e0_e1_n_ho_wo_e2_grid_desc),
        decltype(c_k_n_hop_wop_grid_desc),
        decltype(c_k_n_hop_wop_grid_desc),
        E1,
        E2,
        K2,
        KPerBlock,
        HoPerBlock,
        WoPerBlock,
        E0PerBlock,
        E1PerBlock,
        KPerThread,
        HoPerThread,
        WoPerThread,
        EPerThread,
        ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2,
        ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
        Sequence<2, 3, 0, 1, 4>,
        Sequence<0, 1, 2, 3, 4>,
        4,
        ABlockTransferSrcScalarPerVector_E2,
        ABlockTransferDstScalarPerVector_E2,
        false, // don't move back src coordinate after threadwise copy
        Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>, // E0, E1, N, H0, H1, H2, W0, W1, W2, E2
        9,
        BThreadTransferSrcScalarPerVector_E2,
        false, // don't move back src coordinate after threadwise copy, which will be fused with
               // MoveSrcSliceWindow() to save addr computation
        Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>, // K0, K1, N, H0, H1, H2, W0, W1, W2
        1,
        CThreadTransferDstScalarPerVector_K>;

    const auto a_e0_e1_k0_k1_e2_grid_desc =
        GridwiseGemm::MakeAE0E1K0K1E2GridDescriptor(a_e0_e1_k_e2_grid_desc);
    const auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        GridwiseGemm::MakeBE0E1NH0H1H2W0W1W2E2GridDescriptor(b_e0_e1_n_ho_wo_e2_grid_desc);
    const auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc =
        GridwiseGemm::MakeCK0K1NH0H1H2W0W1W2GridDescriptor(c_k_n_hop_wop_grid_desc);
    const auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        GridwiseGemm::MakeCBlockIdToKNHoWoBlockClusterAdaptor(c_k_n_hop_wop_grid_desc);

    const auto grid_size = (K / KPerBlock) * (Hop / HoPerBlock) * (Wop / WoPerBlock) * N;

    const bool has_main_e0_block_loop = E0 > 1;

    std::cerr << "grid_size = " << grid_size
              << " has_main_e0_block_loop = " << has_main_e0_block_loop << std::endl;

#if 0
    return GemmArguments(GridwiseGemm{},
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                         c_blockid_to_k_n_h_w_block_cluster_adaptor,
                         grid_size,
                         has_main_e0_block_loop);
#else

    static_assert(a_e_k_e2_grid_desc.IsKnownAtCompileTime(), "");
    static_assert(a_e0_e1_k_e2_grid_desc.IsKnownAtCompileTime(), "");

    static_assert(a_e0_e1_k0_k1_e2_grid_desc.IsKnownAtCompileTime(), "");
    static_assert(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.IsKnownAtCompileTime(), "");
    static_assert(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.IsKnownAtCompileTime(), "");
    static_assert(c_blockid_to_k_n_h_w_block_cluster_adaptor.IsKnownAtCompileTime(), "");

    using AGridDesc_E0_E1_K0_K1_E2 = decltype(a_e0_e1_k0_k1_e2_grid_desc);
    using BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 =
        decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc);
    using CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2 = decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
    using CBlockIdToBlockClusterAdaptor_K_N_H_W =
        decltype(c_blockid_to_k_n_h_w_block_cluster_adaptor);

    return GemmArguments<GridwiseGemm,
                         AGridDesc_E0_E1_K0_K1_E2,
                         BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
                         CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
                         CBlockIdToBlockClusterAdaptor_K_N_H_W>(grid_size, has_main_e0_block_loop);
#endif
}

template <typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          typename GridGemmTuningParameters1,
          typename GridGemmTuningParameters2,
          const ck::index_t group_count,
          ck::ActivTypeEnum_t activ_type_>
struct DriverDynamicResizeConcatConvBiasActivForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1
{
    template <typename Conv1Desc, typename Conv2Desc>
    __host__ float Run(Conv1Desc conv1_desc,
                       Conv2Desc conv2_desc,
                       const FloatAB* __restrict__ p_a1_grid,
                       const FloatAB* __restrict__ p_b1_grid,
                       const FloatAB* __restrict__ p_a2_grid,
                       const FloatAB* __restrict__ p_b2_grid,
                       const FloatC* __restrict__ p_bias_grid,
                       FloatC* __restrict__ p_c_grid,
                       const int nrepeat) const
    {

        const auto GemmArg1 =
            MakeGridwiseGemm<FloatAB, FloatAcc, FloatC, GridGemmTuningParameters1>(conv1_desc);

        const auto GemmArg2 =
            MakeGridwiseGemm<FloatAB, FloatAcc, FloatC, GridGemmTuningParameters2>(conv2_desc);

        float ave_time = 0;

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
        {
            const auto kernel =
                kernel_gemm_dlops_v3<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AGridDesc_E0_E1_K0_K1_E2>,
                                     remove_reference_t<BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2>,
                                     remove_reference_t<CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2>,
                                     remove_reference_t<CBlockIdToBlockClusterAdaptor_K_N_H_W>,
                                     true>;

            ave_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              p_a_grid,
                                              p_b_grid,
                                              p_c_grid,
                                              a_e0_e1_k0_k1_e2_grid_desc,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                                              c_blockid_to_k_n_h_w_block_cluster_adaptor);
        }
#elif CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR
        {
            // static_assert(GemmArg1.a_e0_e1_k0_k1_e2_grid_desc.IsKnownAtCompileTime(), "");
            // static_assert(GemmArg1.b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.IsKnownAtCompileTime(),
            //"");
            // static_assert(GemmArg1.c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.IsKnownAtCompileTime(),
            //"");
            // static_assert(
            // GemmArg1.c_blockid_to_k_n_h_w_block_cluster_adaptor.IsKnownAtCompileTime(), "");

            using GridwiseGemm             = decltype(GemmArg1.gridwise_gemm_desc);
            using AGridDesc_E0_E1_K0_K1_E2 = decltype(GemmArg1.a_e0_e1_k0_k1_e2_grid_desc);
            using BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 =
                decltype(GemmArg1.b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc);
            using CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2 =
                decltype(GemmArg1.c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
            using CBlockIdToBlockClusterAdaptor_K_N_H_W =
                decltype(GemmArg1.c_blockid_to_k_n_h_w_block_cluster_adaptor);

            const auto kernel = kernel_gemm_bias_activ_dlops_v3<
                GridwiseGemm,
                FloatAB,
                FloatC,
                remove_reference_t<AGridDesc_E0_E1_K0_K1_E2>,
                remove_reference_t<BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2>,
                remove_reference_t<CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2>,
                remove_reference_t<CBlockIdToBlockClusterAdaptor_K_N_H_W>,
                GemmArg1.has_main_e0_block_loop,
                activ_type_>;

            // ave_time = launch_and_time_kernel(kernel,
            // nrepeat,
            // dim3(grid_size),
            // dim3(BlockSize),
            // 0,
            // p_a_grid,
            // p_b_grid,
            // p_bias_grid,
            // p_c_grid);
        }
#endif
        return ave_time;
    }
};
} // namespace ck
#endif
