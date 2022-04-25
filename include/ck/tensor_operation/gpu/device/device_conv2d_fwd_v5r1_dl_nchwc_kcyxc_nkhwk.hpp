#ifndef DEVICE_CONV2D_FWD_XDL_C_SHUFFLE_NHWC_KYXC_NHWK_HPP
#define DEVICE_CONV2D_FWD_XDL_C_SHUFFLE_NHWC_KYXC_NHWK_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_conv_fwd.hpp"
#include "convolution_forward_specialization.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_dlops_v3.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          ck::index_t BlockSize,
          ck::index_t E1,
          ck::index_t E2,
          ck::index_t K2,
          ck::index_t KPerBlock,
          ck::index_t HoPerBlock,
          ck::index_t WoPerBlock,
          ck::index_t E0PerBlock,
          ck::index_t E1PerBlock,
          ck::index_t KPerThread,
          ck::index_t HoPerThread,
          ck::index_t WoPerThread,
          ck::index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2,
          typename ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
          ck::index_t ABlockTransferSrcScalarPerVector_E2,
          ck::index_t ABlockTransferDstScalarPerVector_E2,
          ck::index_t BThreadTransferSrcScalarPerVector_E2,
          ck::index_t CThreadTransferDstScalarPerVector_K>
struct DeviceConv2dFwdv5r1Xdl_Input_N_C0_Hi_Wi_C1_Weight_K_C0_Y_X_C1_Output_N_K0_Ho_Wo_K1
    : public DeviceConvFwdNCHWc<InElementwiseOperation,
                                WeiElementwiseOperation,
                                OutElementwiseOperation>
{
    using DeviceOp =
        DeviceConv2dFwdv5r1Xdl_Input_N_C0_Hi_Wi_C1_Weight_K_C0_Y_X_C1_Output_N_K0_Ho_Wo_K1;

    using ADataType = InDataType;
    using BDataType = WeiDataType;
    using CDataType = OutDataType;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    // static constexpr index_t NDimSpatial = 2;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    // static constexpr auto K1Number     = Number<K1>{};
    // static constexpr auto GemmK1Number = K1Number;

    static auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(const ck::index_t N,
                                                    const ck::index_t K0,
                                                    const ck::index_t K1,
                                                    const ck::index_t C0,
                                                    const ck::index_t C1,
                                                    std::vector<ck::index_t> input_spatial_lengths,
                                                    std::vector<ck::index_t> filter_spatial_lengths,
                                                    std::vector<ck::index_t> output_spatial_lengths,
                                                    std::vector<ck::index_t> conv_filter_strides,
                                                    std::vector<ck::index_t> conv_filter_dilations,
                                                    std::vector<ck::index_t> input_left_pads,
                                                    std::vector<ck::index_t> input_right_pads)
    {
        using namespace ck;

        const index_t Hi = input_spatial_lengths[0];
        const index_t Wi = input_spatial_lengths[1];

        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        const index_t Y = filter_spatial_lengths[0];
        const index_t X = filter_spatial_lengths[1];

        const index_t K = K0 * K1;

        const auto ConvStrideH = conv_filter_strides[I0];
        const auto ConvStrideW = conv_filter_strides[I1];

        const auto ConvDilationH = conv_filter_dilations[I0];
        const auto ConvDilationW = conv_filter_dilations[I1];

        const auto Hop = (Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock;
        const auto Wop = (Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock;

        const auto OutRightPadH = Hop - Ho;
        const auto OutRightPadW = Wop - Wo;

        const auto InLeftPadH = input_left_pads[I0];
        const auto InLeftPadW = input_left_pads[I1];

        const auto InRightPadH = input_right_pads[I0] + OutRightPadH * ConvStrideH;
        const auto InRightPadW = input_right_pads[I1] + OutRightPadW * ConvStrideW;

        const auto E0 = (C0 * Y * X) / E1;

        // constexpr auto E1_ = Number<E1>{};
        // constexpr auto E2_ = Number<E2>{};
        // constexpr auto K2_ = Number<K2>{};

        if((C0 * Y * X) % (E1 * E0PerBlock) != 0)
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        if(E2 != C1)
        {
            throw std::runtime_error("wrong! E2 != C1");
        }

        // weight tensor
        const auto a_e_k_e2_grid_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(K, C0 * Y * X, E2)),
            make_tuple(make_pass_through_transform(K),
                       make_pass_through_transform(C0 * Y * X),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}, Sequence<2>{}));

        const auto a_e0_e1_k_e2_grid_desc =
            transform_tensor_descriptor(a_e_k_e2_grid_desc,
                                        make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                                                   make_pass_through_transform(K),
                                                   make_pass_through_transform(E2)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

        // input tensor
        const auto in_n_c0_hip_wip_e2_global_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi, E2)),
            make_tuple(make_pass_through_transform(N),
                       make_pass_through_transform(C0),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        const auto in_n_c0_y_ho_x_wo_e2_global_desc = transform_tensor_descriptor(
            in_n_c0_hip_wip_e2_global_desc,
            make_tuple(
                make_pass_through_transform(N),
                make_pass_through_transform(C0),
                make_embed_transform(make_tuple(Y, Hop), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wop), make_tuple(ConvDilationW, ConvStrideW)),
                make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}, Sequence<6>{}));

        const auto in_e_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
            in_n_c0_y_ho_x_wo_e2_global_desc,
            make_tuple(make_merge_transform(make_tuple(C0, Y, X)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Hop),
                       make_pass_through_transform(Wop),
                       make_pass_through_transform(E2)),
            make_tuple(
                Sequence<1, 2, 4>{}, Sequence<0>{}, Sequence<3>{}, Sequence<5>{}, Sequence<6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        const auto b_e0_e1_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
            in_e_n_ho_wo_e2_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Hop),
                       make_pass_through_transform(Wop),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}));

        // output tensor
        const auto c_k_n_hop_wop_grid_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1)),
            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_pad_transform(Ho, I0, OutRightPadH),
                       make_pad_transform(Wo, I0, OutRightPadW)),
            make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        return make_tuple(
            a_e0_e1_k_e2_grid_desc, b_e0_e1_n_ho_wo_e2_grid_desc, c_k_n_hop_wop_grid_desc);
    }

    using ABCGridDescs = decltype(MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        1, 1, 1, 1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}));

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmDlops_km_kn_mn_v3<
        BlockSize,
        ABDataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType, // TODO: Add ShuffleType for DeviceConv2d
        InMemoryDataOperationEnum::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        CGridDesc_M_N,
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

    // Argument
    struct Argument : public BaseArgument
    {
        template <typename InputSpatialLengths,
                  typename FilterSpatialLengths,
                  typename OutputSpatialLengths,
                  typename ConvFilterStrides,
                  typename ConvFilterDilations,
                  typename InputLeftPads,
                  typename InputRightPads>
        Argument(const InDataType* p_in_grid,
                 const WeiDataType* p_wei_grid,
                 OutDataType* p_out_grid,
                 ck::index_t N,
                 ck::index_t K0,
                 ck::index_t K1,
                 ck::index_t C0,
                 ck::index_t C1,
                 const InputSpatialLengths input_spatial_lengths,
                 const FilterSpatialLengths filter_spatial_lengths,
                 const OutputSpatialLengths output_spatial_lengths,
                 const ConvFilterStrides conv_filter_strides,
                 const ConvFilterDilations conv_filter_dilations,
                 const InputLeftPads input_left_pads,
                 const InputRightPads input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : p_a_grid_{p_in_grid},
              p_b_grid_{p_wei_grid},
              p_c_grid_{p_out_grid},
              a_e0_e1_k_e2_grid_desc_{},
              b_e0_e1_n_ho_wo_e2_grid_desc_{},
              c_k_n_hop_wop_grid_desc_{},
              // c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_{},
              // block_2_ctile_map_{},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        // Conv_N_{N},
        // Conv_K0_{K0},
        // Conv_K1_{K1},
        // Conv_C0_{C0},
        // Conv_C1_{C1},
        // input_spatial_lengths_{input_spatial_lengths},
        // filter_spatial_lengths_{filter_spatial_lengths},
        // output_spatial_lengths_{output_spatial_lengths},
        // conv_filter_strides_{conv_filter_strides},
        // conv_filter_dilations_{conv_filter_dilations},
        // input_left_pads_{input_left_pads},
        // input_right_pads_{input_right_pads}
        {
            const auto descs =
                DeviceOp::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(N,
                                                                          K0,
                                                                          K1,
                                                                          C0,
                                                                          C1,
                                                                          input_spatial_lengths,
                                                                          filter_spatial_lengths,
                                                                          output_spatial_lengths,
                                                                          conv_filter_strides,
                                                                          conv_filter_dilations,
                                                                          input_left_pads,
                                                                          input_right_pads);

            a_e0_e1_k_e2_grid_desc_       = descs[I0];
            b_e0_e1_n_ho_wo_e2_grid_desc_ = descs[I1];
            c_k_n_hop_wop_grid_desc_      = descs[I2];

            // if(GridwiseGemm::CheckValidity(
            // a_e0_e1_k_e2_grid_desc_, b_e0_e1_n_ho_wo_e2_grid_desc_, c_k_n_hop_wop_grid_desc_,
            // M01_, N01_))
            //{
            // c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_ =
            // GridwiseGemm::
            // MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
            // c_k_n_hop_wop_grid_desc_);

            // block_2_ctile_map_ =
            // GridwiseGemm::MakeDefaultBlock2CTileMap(c_k_n_hop_wop_grid_desc_, M01, N01);
            //}
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_e0_e1_k_e2_grid_desc_;
        BGridDesc_K0_N_K1 b_e0_e1_n_ho_wo_e2_grid_desc_;
        CGridDesc_M_N c_k_n_hop_wop_grid_desc_;
        // typename GridwiseGemm::
        // CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
        // c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_;
        // typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
        // for checking IsSupportedArgument()
        index_t Conv_N_;
        index_t Conv_K0_;
        index_t Conv_K1_;
        index_t Conv_C0_;
        index_t Conv_C1_;
        std::vector<index_t> input_spatial_lengths_;
        std::vector<index_t> filter_spatial_lengths_;
        std::vector<index_t> output_spatial_lengths_;
        std::vector<index_t> conv_filter_strides_;
        std::vector<index_t> conv_filter_dilations_;
        std::vector<index_t> input_left_pads_;
        std::vector<index_t> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            // if(!GridwiseGemm::CheckValidity(arg.a_e0_e1_k_e2_grid_desc_,
            // arg.b_e0_e1_n_ho_wo_e2_grid_desc_,
            // arg.c_k_n_hop_wop_grid_desc_,
            // arg.M01_,
            // arg.N01_))
            //{
            // throw std::runtime_error(
            //"wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v3r1 has invalid setting");
            //}

            const auto a_e0_e1_k0_k1_e2_grid_desc =
                GridwiseGemm::MakeAE0E1K0K1E2GridDescriptor(arg.a_e0_e1_k_e2_grid_desc_);
            const auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
                GridwiseGemm::MakeBE0E1NH0H1H2W0W1W2E2GridDescriptor(
                    arg.b_e0_e1_n_ho_wo_e2_grid_desc_);
            const auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc =
                GridwiseGemm::MakeCK0K1NH0H1H2W0W1W2GridDescriptor(arg.c_k_n_hop_wop_grid_desc_);

            const auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
                GridwiseGemm::MakeCBlockIdToKNHoWoBlockClusterAdaptor(arg.c_k_n_hop_wop_grid_desc_);

            using AGridDesc_E0_E1_K0_K1_E2 = decltype(a_e0_e1_k0_k1_e2_grid_desc);
            using BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 =
                decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc);
            using CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2 =
                decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

            using CBlockIdToBlockClusterAdaptor_K_N_H_W =
                decltype(c_blockid_to_k_n_h_w_block_cluster_adaptor);

            // const auto grid_size = (K / KPerBlock) * (Hop / HoPerBlock) * (Wop / WoPerBlock) * N;
            const index_t grid_size = GridwiseGemm::CalculateGridSize(arg.c_k_n_hop_wop_grid_desc_);

            // const auto K0 = arg.a_e0_e1_k_e2_grid_desc_.GetLength(I0);

            // const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

            const auto kernel =
                kernel_gemm_dlops_v3<GridwiseGemm,
                                     ADataType,
                                     CDataType,
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
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_c_grid_,
                                              a_e0_e1_k0_k1_e2_grid_desc,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                                              c_blockid_to_k_n_h_w_block_cluster_adaptor);
            return ave_time;
        }

        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        return true;
        // vector load A/B matrix from global memory
        // if(!(ABlockTransferSrcVectorDim == 2 && BBlockTransferSrcVectorDim == 2 &&
        // arg.Conv_C_ % ABlockTransferSrcScalarPerVector == 0 &&
        // arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
        //{
        // return false;
        //}

        //// vector store C matrix into global memory
        // if(!(arg.Conv_K_ % CBlockTransferScalarPerVector_NWaveNPerXdl == 0))
        //{
        // return false;
        //}

        // Gridwise GEMM size
        // return GridwiseGemm::CheckValidity(arg.a_e0_e1_k_e2_grid_desc_,
        // arg.b_e0_e1_n_ho_wo_e2_grid_desc_,
        // arg.c_k_n_hop_wop_grid_desc_,
        // arg.M01_,
        // arg.N01_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in_grid,
                             const WeiDataType* p_wei_grid,
                             OutDataType* p_out_grid,
                             ck::index_t N,
                             ck::index_t K0,
                             ck::index_t K1,
                             ck::index_t C0,
                             ck::index_t C1,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> filter_spatial_lengths,
                             std::vector<ck::index_t> output_spatial_lengths,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        N,
                        K0,
                        K1,
                        C0,
                        C1,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        const void* p_wei_grid,
                        void* p_out_grid,
                        ck::index_t N,
                        ck::index_t K0,
                        ck::index_t K1,
                        ck::index_t C0,
                        ck::index_t C1,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<const WeiDataType*>(p_wei_grid),
                                          static_cast<OutDataType*>(p_out_grid),
                                          N,
                                          K0,
                                          K1,
                                          C0,
                                          C1,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          output_spatial_lengths,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        //str << "DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K"
            //<< "<"
            //<< BlockSize << ", "
            //<< MPerBlock << ", "
            //<< NPerBlock << ", "
            //<< K0PerBlock << ", "
            //<< getConvFwdSpecializationStr(ConvForwardSpecialization)
            //<< ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
