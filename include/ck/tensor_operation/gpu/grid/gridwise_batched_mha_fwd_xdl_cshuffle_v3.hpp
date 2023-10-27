// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_softmax.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_dropout.hpp"

namespace ck {

template <typename FloatAB,
          typename ZDataType,
          typename FloatGemm,
          typename FloatGemmAcc,
          typename FloatCShuffle,
          typename FloatC,
          typename FloatLSE,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename B1GridDesc_BK0_N_BK1,
          typename CGridDesc_M_N,
          typename ZGridDesc_M_N,
          typename LSEGridDesc_M,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t Gemm1NPerBlock,
          index_t Gemm1KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t B1K1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          index_t Gemm1NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t BBlockLdsExtraN,
          typename B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_BK1,
          bool B1ThreadTransferSrcResetCoordinateAfterRun,
          index_t B1BlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched,
          bool PadN,
          bool MaskOutUpperTriangle,
          bool Deterministic,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseBatchedMultiheadAttentionForward_Xdl_CShuffle
{
    static_assert(LoopSched == LoopScheduler::Default,
                  "Non-default loop scheduler is currently not supported");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    static constexpr auto WaveSize = 64;

    // K1 should be Number<...>
    // Gemm0
    static constexpr auto AK0 = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0 = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1 = Number<AK1Value>{};
    static constexpr auto BK1 = Number<BK1Value>{};

    static constexpr auto Gemm0MWaves = MPerBlock / (MPerXdl * MXdlPerWave);
    static constexpr auto Gemm0NWaves = NPerBlock / (NPerXdl * NXdlPerWave);

    // Gemm1
    static constexpr auto B1K0 = Number<Gemm1KPerBlock / B1K1Value>{};
    static constexpr auto B1K1 = Number<B1K1Value>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = remove_cvref_t<decltype(
        GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;

    // C desc for source in gridwise copy
    __host__ __device__ static constexpr auto MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(
        const ZGridDesc_M_N& z_grid_desc_m_n) ////=> for z use
    {
        const auto M = z_grid_desc_m_n.GetLength(I0);
        const auto N = z_grid_desc_m_n.GetLength(I1);

        constexpr auto mfma = MfmaSelector<FloatGemm, MPerXdl, NPerXdl>::selected_mfma;
        constexpr auto N3   = mfma.num_groups_per_blk;
        constexpr auto N4   = mfma.num_input_blks;
        constexpr auto N5   = mfma.group_size;
        return transform_tensor_descriptor(
            z_grid_desc_m_n,
            make_tuple(make_unmerge_transform(
                           make_tuple(M / MPerBlock, MXdlPerWave, Gemm0MWaves, MPerXdl)),
                       make_unmerge_transform(
                           make_tuple(N / NPerBlock, NXdlPerWave, Gemm0NWaves, N3, N4, N5))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7, 8, 9>{}));
    }

    __host__ __device__ static constexpr auto GetZShuffleBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto mfma = MfmaSelector<FloatGemm, MPerXdl, NPerXdl>::selected_mfma;
        constexpr auto M0   = MXdlPerWave;
        constexpr auto M1   = Gemm0MWaves;
        constexpr auto N1   = Gemm0NWaves;
        constexpr auto M2   = MPerXdl;
        constexpr auto N2   = mfma.num_groups_per_blk;
        constexpr auto N3   = mfma.num_input_blks;
        constexpr auto N4   = mfma.group_size;

        constexpr auto z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            make_naive_tensor_descriptor_packed(make_tuple(M0, I1, M1, N1, M2, N2, N3, N4));

        return z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4;
    }

    __host__ __device__ static constexpr auto GetPaddedSize(const index_t size)
    {
        constexpr auto mfma       = MfmaSelector<FloatGemm, MPerXdl, NPerXdl>::selected_mfma;
        constexpr auto group_size = mfma.group_size;
        return math::integer_divide_ceil(size, group_size) * group_size;
    }

    __device__ static auto GetGemm0WaveIdx()
    {
        const index_t thread_id = get_thread_local_1d_id();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(Gemm0MWaves, Gemm0NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto GetGemm0WaveMNIdx(const index_t thread_id)
    {
        constexpr auto wave_threadid_to_mn_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(WaveSize / MPerXdl, MPerXdl))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return wave_threadid_to_mn_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, MWaves, MPerXdl>(
            ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, NWaves, NPerXdl>(
            BBlockDesc_BK0_N_BK1{});
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, 1, 1>(ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t Gemm1NWaves = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);
        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm1NXdlPerWave, Gemm1NWaves, NPerXdl>(
            BBlockDesc_BK0_N_BK1{});
    }

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0, Number<MPerBlock>{}, AK1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    __host__ __device__ static constexpr auto GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B1 matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(B1K0, Number<Gemm1NPerBlock>{}, B1K1),
            make_tuple(Number<Gemm1NPerBlock + B1BlockLdsExtraN>{} * B1K1, B1K1, I1));
    }

    __host__ __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        const index_t gemm0_bytes_end = (SharedMemTrait::a_block_space_size_aligned +
                                         SharedMemTrait::b_block_space_size_aligned) *
                                        sizeof(FloatGemm);
        const index_t gemm1_bytes_end =
            (SharedMemTrait::b1_block_space_offset + SharedMemTrait::b1_block_space_size_aligned) *
            sizeof(FloatGemm);
        const index_t softmax_bytes_end = (SharedMemTrait::reduction_space_offset +
                                           SharedMemTrait::reduction_space_size_aligned) *
                                          sizeof(FloatGemmAcc);
        const index_t c_block_bytes_end =
            SharedMemTrait::c_block_space_size * sizeof(FloatCShuffle);

        const index_t z_block_bytes_end =
            SharedMemTrait::z_shuffle_block_space_size * sizeof(ushort);

        return math::max(gemm0_bytes_end,
                         gemm1_bytes_end,
                         softmax_bytes_end,
                         c_block_bytes_end,
                         z_block_bytes_end);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                  const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                  const B1GridDesc_BK0_N_BK1& b1_grid_desc_bk0_n_bk1,
                  const CGridDesc_M_N& c_grid_desc_m_n,
                  const Block2CTileMap& block_2_ctile_map)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M = a_grid_desc_ak0_m_ak1.GetLength(I1);
        const auto N = b_grid_desc_bk0_n_bk1.GetLength(I1);
        const auto K = a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2);
        const auto Gemm1N = b1_grid_desc_bk0_n_bk1.GetLength(I1);

        // if(Gemm1N != K)
        // {
        //     std::cout << "SizeK must be equal to SizeO (equal attention head size)" << '\n';
        //     return false;
        // }

        if(!(M == c_grid_desc_m_n.GetLength(I0) && Gemm1N == c_grid_desc_m_n.GetLength(I1)))
        {
            return false;
        }

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0 &&
             Gemm1N % Gemm1NPerBlock == 0))
        {
            return false;
        }

        // check gemm0 gridwise gemm pipeline
        const auto num_gemm0_k_loop = K / KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm0_k_loop))
        {
            return false;
        }

        // check gemm1 gridwise gemm pipeline
        if(!(NPerBlock % Gemm1KPerBlock == 0))
        {
            return false;
        }

        const auto num_gemm1_k_inner_loop = NPerBlock / Gemm1KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm1_k_inner_loop))
        {
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(c_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / Gemm1NPerBlock;

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    __host__ __device__ static constexpr auto
    MakeLSEGridDescriptor_MBlock_MRepeat_NWave_MPerXdl(const LSEGridDesc_M& lse_grid_desc_m)
    {
        const index_t M         = lse_grid_desc_m.GetLength(I0);
        const index_t MBlock    = M / MPerBlock;
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);

        const auto lse_grid_desc_mblock_mrepeat_mwave_mperxdl = transform_tensor_descriptor(
            lse_grid_desc_m,
            make_tuple(make_unmerge_transform(
                make_tuple(MBlock, Number<MXdlPerWave>{}, MWave, Number<MPerXdl>{}))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2, 3>{}));

        return lse_grid_desc_mblock_mrepeat_mwave_mperxdl;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, Gemm1NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }

    using CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}))>;

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}))>;

    using ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5 = remove_cvref_t<decltype(
        MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(ZGridDesc_M_N{}))>;

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        static constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr auto b1_block_desc_bk0_n_bk1 =
            GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        static constexpr auto max_lds_align = math::lcm(math::lcm(AK1, BK1), B1K1);

        static constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b1_block_space_size_aligned = math::integer_least_multiple(
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        static constexpr auto a_block_space_offset  = 0;
        static constexpr auto b_block_space_offset  = a_block_space_size_aligned.value;
        static constexpr auto b1_block_space_offset = 0;

        // LDS allocation for reduction
        static constexpr index_t reduction_space_size_aligned =
            math::integer_least_multiple(BlockSize, max_lds_align);

        static constexpr auto reduction_space_offset = 0;

        // LDS allocation for C shuffle in LDS
        static constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();
        static constexpr auto c_block_space_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        // LDS allocation for Z shuffle in LDS
        static constexpr auto z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            GetZShuffleBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();
        static constexpr auto z_shuffle_block_space_size =
            z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetElementSpaceSize();
    };

    template <bool HasMainKBlockLoop,
              bool IsDropout,
              bool IsLseStoring,
              typename Block2CTileMap,
              typename C0MatrixMask>
    __device__ static void Run(const FloatAB* __restrict__ p_a_grid,
                               const FloatAB* __restrict__ p_b_grid,
                               const FloatAB* __restrict__ p_b1_grid,
                               FloatC* __restrict__ p_c_grid,
                               ZDataType* __restrict__ p_z_grid,
                               FloatLSE* __restrict__ p_lse_grid,
                               void* __restrict__ p_shared,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const AccElementwiseOperation& acc_element_op,
                               const B1ElementwiseOperation& b1_element_op,
                               const CElementwiseOperation& c_element_op,
                               const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                               const B1GridDesc_BK0_N_BK1& b1_grid_desc_bk0_n_bk1,
                               const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5&
                                   z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                               const LSEGridDesc_M& lse_grid_desc_m,
                               const Block2CTileMap& block_2_ctile_map,
                               const C0MatrixMask& c0_matrix_mask,
                               const ushort p_dropout_in_16bits,
                               FloatGemmAcc p_dropout_rescale,
                               ck::philox& ph,
                               const index_t z_random_matrix_offset,
                               const index_t raw_n_padded,
                               const index_t block_idx_m)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        const auto b1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b1_grid, b1_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());
        auto lse_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_lse_grid, lse_grid_desc_m.GetElementSpaceSize());

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_work_idx_m = Deterministic ? block_idx_m : block_work_idx[I0];

        // HACK: this force m/gemm1_n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx_m * MPerBlock);

        const index_t gemm1_n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * Gemm1NPerBlock);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        //
        // set up Gemm0
        //

        // A matrix blockwise copy
        auto q_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatGemm,
                                                decltype(a_grid_desc_ak0_m_ak1),
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>(
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto k_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatGemm,
                                                decltype(b_grid_desc_bk0_n_bk1),
                                                decltype(b_block_desc_bk0_n_bk1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                BBlockTransferSrcVectorDim,
                                                2,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>(
                b_grid_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0), // will loop over GemmN dimension
                b_element_op,
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // Fused Gemm+Gemm pipeline
        // for n in N0:
        //   for k in K0:
        //     acc[m][n] += A[m][k] * B0[k][n]
        //   acc1[m][o] += acc[m][n] * B1[n][o]

        // sanity check
        constexpr index_t KPack =
            math::max(math::lcm(AK1, BK1),
                      MfmaSelector<FloatGemm, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        auto qk_blockwise_gemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            FloatGemm,
            FloatGemmAcc,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            decltype(MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(a_block_desc_ak0_m_ak1)),
            decltype(MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(b_block_desc_bk0_n_bk1)),
            MPerBlock,
            NPerBlock,
            KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            true>{}; // TransposeC

        auto acc_thread_buf = qk_blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        auto q_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatGemm*>(p_shared) + SharedMemTrait::a_block_space_offset,
            a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto k_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatGemm*>(p_shared) + SharedMemTrait::b_block_space_offset,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto q_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        constexpr auto k_block_slice_copy_step = make_multi_index(KPerBlock / BK1, 0, 0);
        const auto q_block_reset_copy_step =
            make_multi_index(-a_grid_desc_ak0_m_ak1.GetLength(I0), 0, 0);
        const auto k_block_reset_copy_step =
            make_multi_index(-b_grid_desc_bk0_n_bk1.GetLength(I0), NPerBlock, 0);

        // gridwise GEMM pipeline
        // Only supports LoopScheduler::Default
        //const auto gridwise_gemm_pipeline = GridwiseGemmPipeline_Selector<PipelineVer,
        //                                                                  NumGemmKPrefetchStage,
        //                                                                  LoopScheduler::Default>();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        //
        // set up Gemm1
        //

        // Acc matrix threadwise copy: AccVGPR to VGPR and downcast to XDL input data type
        constexpr auto acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            qk_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        constexpr auto m0 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
        constexpr auto n0 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
        constexpr auto m1 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
        constexpr auto n1 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
        constexpr auto m2 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
        constexpr auto n2 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
        constexpr auto n3 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
        constexpr auto n4 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);

        constexpr auto b1_block_slice_copy_step = make_multi_index(Gemm1KPerBlock / B1K1, 0, 0);

        // acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 to acc_thread_desc_k0_m_k1
        // n0_n1_n2_n3 -> k0
        // m0_m1_m2 -> m
        // n4 -> k1
        // NOTE: had to use merge_v3 or will spit out compilation errors
        constexpr auto acc_thread_desc_k0_m_k1 = transform_tensor_descriptor(
            acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(n0, n1, n2, n3)),
                       make_merge_transform_v3_division_mod(make_tuple(m0, m1, m2)),
                       make_pass_through_transform(n4)),
            make_tuple(Sequence<1, 3, 5, 6>{}, Sequence<0, 2, 4>{}, Sequence<7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // A1 matrix in AccVGPR
        // N2 num_groups_per_blk, N3 num_input_blks, N4 group_size
        constexpr auto AccN3 =
            qk_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLength(I6);
        constexpr auto AccM2 =
            qk_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLength(I4);

        constexpr auto A1ThreadSlice_K0_M_K1 =
            make_tuple(Number<Gemm1KPerBlock / n4 / AccN3>{}, Number<m0 * m1 * m2>{}, Number<n4>{});

        constexpr auto A1ThreadSliceK0        = A1ThreadSlice_K0_M_K1[I0];
        constexpr auto A1ThreadSliceM         = A1ThreadSlice_K0_M_K1[I1];
        constexpr auto A1ThreadSliceK1        = A1ThreadSlice_K0_M_K1[I2];
        constexpr auto a1_thread_desc_k0_m_k1 = make_naive_tensor_descriptor(
            A1ThreadSlice_K0_M_K1,
            make_tuple(A1ThreadSliceM * A1ThreadSliceK1, A1ThreadSliceK1, I1));

        // B1 matrix in LDS memory, dst of blockwise copy
        constexpr auto b1_block_desc_bk0_n_bk1 = GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // A1 matrix blockwise copy
        auto a1_blockwise_copy = ThreadwiseTensorSliceTransfer_StaticToStatic<
            FloatGemmAcc,
            FloatGemm,
            decltype(acc_thread_desc_k0_m_k1),
            decltype(a1_thread_desc_k0_m_k1),
            tensor_operation::element_wise::PassThrough,
            Sequence<A1ThreadSliceK0, A1ThreadSliceM, A1ThreadSliceK1>,
            Sequence<1, 0, 2>,
            2,
            n4>{tensor_operation::element_wise::PassThrough{}};

        // B1 matrix blockwise copy
        auto b1_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<B1K0, Gemm1NPerBlock, B1K1>,
                                                B1BlockTransferThreadClusterLengths_BK0_N_BK1,
                                                B1BlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatGemm,
                                                decltype(b1_grid_desc_bk0_n_bk1),
                                                decltype(b1_block_desc_bk0_n_bk1),
                                                B1BlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                B1BlockTransferSrcVectorDim,
                                                2,
                                                B1BlockTransferSrcScalarPerVector,
                                                B1BlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                B1ThreadTransferSrcResetCoordinateAfterRun,
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>(
                b1_grid_desc_bk0_n_bk1,
                make_multi_index(0, gemm1_n_block_data_idx_on_grid, 0),
                b1_element_op,
                b1_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        auto a1_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatGemm>(
            a1_thread_desc_k0_m_k1.GetElementSpaceSize());

        // reuse LDS space for gemm0's b_block_buf
        auto b1_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatGemm*>(p_shared) + SharedMemTrait::b1_block_space_offset,
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize());

        // selected_mfma.group_size or B1K1 <= Gemm1KPack <= selected_mfma.group_size
        // selected_mfma.k_per_blk <= Gemm1KPack
        //
        // Following similar rationale behind Gemm0KPack, let Gemm1KPack be the lowest common
        // multiples of A1K1 (predetermined by selected_mfma.group_size) and B1K1. But in this case
        // Gemm1KPack can't be higher than A1K1 itself because A1 matrix is distributed in VGPRs
        // with 'group_size' amount of contiguous elements. Having Gemm1KPack greater than A1K1 will
        // cause mismatch in summation index for example c[0:7] = a1[[0:3, 8:11]] * b1[0:7].
        // therefore we may just as well assign Gemm1KPack = group_size
        constexpr index_t Gemm1KPack =
            MfmaSelector<FloatGemm, MPerXdl, NPerXdl>::selected_mfma.group_size;

        auto gemm1_blockwise_gemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            FloatGemm,
            FloatGemmAcc,
            decltype(a1_thread_desc_k0_m_k1),
            decltype(b1_block_desc_bk0_n_bk1),
            decltype(MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(a1_thread_desc_k0_m_k1)),
            decltype(MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(b1_block_desc_bk0_n_bk1)),
            MPerBlock,
            Gemm1NPerBlock,
            Gemm1KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            Gemm1NXdlPerWave,
            Gemm1KPack,
            true,       // TransposeC
            Gemm1KPack, // AMmaKStride
            Gemm1KPack * XdlopsGemm<FloatGemm, MPerXdl, NPerXdl, Gemm1KPack, false>{}.K0PerXdlops>{
            // BMmaKStride
            make_tuple(0, 0, 0, 0)}; // A_origin

        auto acc1_thread_buf = gemm1_blockwise_gemm.GetCThreadBuffer();

        //
        // Blockwise softmax
        //
        auto workspace_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatGemmAcc*>(p_shared) + SharedMemTrait::reduction_space_offset,
            SharedMemTrait::reduction_space_size_aligned);

        // get acc0 8D thread cluster
        constexpr auto thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4 =
            qk_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths() /
            qk_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();
        constexpr auto tm0 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I0);
        constexpr auto tn0 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I1);
        constexpr auto tm1 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I2);
        constexpr auto tn1 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I3);
        constexpr auto tm2 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I4);
        constexpr auto tn2 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I5);
        constexpr auto tn3 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I6);
        constexpr auto tn4 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I7);

        // get acc0 thread map
        constexpr auto m0_n_m1_to_m_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(tm0 * tm1, tm2)),
                       make_pass_through_transform(I1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        constexpr auto threadid_to_m0_n_m1_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_merge_transform(make_tuple(tm0 * tm1, tn0 * tn1 * tn2 * tn3 * tn4, tm2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));
        const auto threadid_to_m_n_thread_cluster_adaptor =
            chain_tensor_adaptors(m0_n_m1_to_m_n_adaptor, threadid_to_m0_n_m1_adaptor);

        // get acc0 2D thread cluster & 2D thread slice
        constexpr auto thread_cluster_desc_m_n = make_naive_tensor_descriptor_packed(
            make_tuple(tm0 * tm1 * tm2, tn0 * tn1 * tn2 * tn3 * tn4));
        constexpr auto thread_slice_desc_m_n =
            make_naive_tensor_descriptor_packed(make_tuple(m0 * m1 * m2, n0 * n1 * n2 * n3 * n4));

        auto blockwise_softmax = BlockwiseSoftmax_v1<BlockSize,
                                                  FloatGemmAcc,
                                                  decltype(threadid_to_m_n_thread_cluster_adaptor),
                                                  decltype(thread_cluster_desc_m_n),
                                                  decltype(thread_slice_desc_m_n)>{};

        auto blockwise_dropout = BlockwiseDropout<FloatGemmAcc, decltype(thread_slice_desc_m_n)>{
            p_dropout_in_16bits, p_dropout_rescale};

        const index_t num_gemm1_k_block_outer_loop =
            b_grid_desc_bk0_n_bk1.GetLength(I1) / NPerBlock;
        constexpr index_t num_gemm1_k_block_inner_loop = NPerBlock / Gemm1KPerBlock;

        // Initialize C
        StaticBuffer<AddressSpaceEnum::Vgpr, FloatGemmAcc, acc1_thread_buf.Size(), true>
            c_thread_buf;
        c_thread_buf.Clear();

        // Initialize running sum and max of exponentiating row vectors
        using SoftmaxBuf = typename decltype(blockwise_softmax)::BufferType;
        SoftmaxBuf running_sum, running_sum_new, running_max, running_max_new;
        running_sum     = 0;
        running_sum_new = 0;
        running_max     = NumericLimits<FloatGemmAcc>::Lowest();
        running_max_new = NumericLimits<FloatGemmAcc>::Lowest();

        auto lse_grid_desc_mblock_mrepeat_mwave_mperxdl =
            MakeLSEGridDescriptor_MBlock_MRepeat_NWave_MPerXdl(lse_grid_desc_m);

        constexpr auto lse_thread_desc_mblock_mrepeat_mwave_mperxdl =
            make_naive_tensor_descriptor_packed(make_tuple(I1, m0, m1, m2));

        auto lse_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatLSE>(
            lse_thread_desc_mblock_mrepeat_mwave_mperxdl.GetElementSpaceSize());

        auto acc0_thread_origin = qk_blockwise_gemm.CalculateCThreadOriginDataIndex8D(
            Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});
        auto lse_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            FloatLSE,
            decltype(lse_thread_desc_mblock_mrepeat_mwave_mperxdl),
            decltype(lse_grid_desc_mblock_mrepeat_mwave_mperxdl),
            ck::tensor_operation::element_wise::PassThrough,
            Sequence<1, 1, 1, 1>,
            Sequence<0, 1, 2, 3>,
            3,
            1,
            InMemoryDataOperationEnum::Set,
            1,
            false>{lse_grid_desc_mblock_mrepeat_mwave_mperxdl,
                   make_multi_index(block_work_idx_m,        // mblock
                                    0,                       // mrepeat
                                    acc0_thread_origin[I2],  // mwave
                                    acc0_thread_origin[I4]), // mperxdl
                   ck::tensor_operation::element_wise::PassThrough{}};

        // gemm1 K loop
        index_t gemm1_k_block_outer_index = 0;
	index_t k_block_start = 0;
        index_t q_block_start = m_block_data_idx_on_grid;


        // z is random number matrix for dropout verify
        //
        // z vgpr copy to global
        //
        // z matrix threadwise desc
        constexpr auto z_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =   // for blockwise copy
            make_naive_tensor_descriptor_packed(make_tuple(m0,   // MRepeat
                                                           I1,   // NRepeat
                                                           m1,   // MWaveId
                                                           n1,   // NWaveId
                                                           m2,   // MPerXdl
                                                           n2,   // NGroupNum
                                                           n3,   // NInputNum
                                                           n4)); // RegisterNum

        constexpr auto z_shuffle_thread_desc_m0_n0_m1_n1_m2_n2_n3_m3_n4 = // for blockwise copy
            make_naive_tensor_descriptor_packed(make_tuple(m0,            // MRepeat
                                                           I1,            // NRepeat
                                                           m1,            // MWaveId
                                                           n1,            // NWaveId
                                                           m2,            // MPerXdl
                                                           n2,            // NGroupNum
                                                           n3,            // NInputNum
                                                           n4,            // RegisterNum
                                                           I1));          // I1

        constexpr auto z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,   // MBlockId
                                                           I1,   // NBlockId
                                                           m0,   // MRepeat
                                                           I1,   // NRepeat
                                                           m1,   // MWaveId
                                                           n1,   // NWaveId
                                                           m2,   // MPerXdl
                                                           n2,   // NGroupNum
                                                           n3,   // NInputNum
                                                           n4)); // RegisterNum

        constexpr auto z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            GetZShuffleBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        constexpr auto ZM0 = z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
        constexpr auto ZN0 = z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
        constexpr auto ZM1 = z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
        constexpr auto ZN1 = z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
        constexpr auto ZM2 = z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
        constexpr auto ZN2 = z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
        constexpr auto ZN3 = z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
        constexpr auto ZN4 = z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);

        constexpr auto z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_m3_n4 =
            transform_tensor_descriptor(
                z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                make_tuple(make_pass_through_transform(ZM0),
                           make_pass_through_transform(ZN0),
                           make_pass_through_transform(ZM1),
                           make_pass_through_transform(ZN1),
                           make_unmerge_transform(make_tuple(ZM2 / ZN4, ZN4)),
                           make_pass_through_transform(ZN2),
                           make_pass_through_transform(ZN3),
                           make_pass_through_transform(ZN4)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{},
                           Sequence<6>{},
                           Sequence<7>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4, 7>{},
                           Sequence<5>{},
                           Sequence<6>{},
                           Sequence<8>{}));

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     ushort,
                     z_shuffle_thread_desc_m0_n0_m1_n1_m2_n2_n3_m3_n4.GetElementSpaceSize(),
                     true>
            z_tensor_buffer;
        z_tensor_buffer.Clear();

        auto z_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_z_grid, z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5.GetElementSpaceSize());

        auto z_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ushort*>(p_shared),
            z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetElementSpaceSize());

        const auto wave_id     = GetGemm0WaveIdx();
        const auto wave_m_n_id = GetGemm0WaveMNIdx(wave_id[I2]); // I2: 0~63

        auto z_tmp_thread_copy_vgpr_to_lds = ThreadwiseTensorSliceTransfer_v1r3<
            ushort,
            ushort,
            decltype(z_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
            decltype(z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
            tensor_operation::element_wise::PassThrough,
            Sequence<m0,  // MRepeat
                     I1,  // NRepeat
                     m1,  // MWaveId
                     n1,  // NWaveId
                     m2,  // MPerXdl
                     n2,  // NGroupNum
                     n3,  // NInputNum
                     n4>, // RegisterNum
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
            7, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>{z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                  make_multi_index(0,               // MRepeat
                                   0,               // NRepeat
                                   wave_id[I0],     // MWaveId
                                   wave_id[I1],     // NWaveId
                                   wave_m_n_id[I1], // MPerXdl
                                   0,               // NGroupIndex
                                   wave_m_n_id[I0], // NInputIndex
                                   0),
                  tensor_operation::element_wise::PassThrough{}};

        auto z_shuffle_thread_copy_lds_to_vgpr = ThreadwiseTensorSliceTransfer_v2<
            ushort,
            ushort,
            decltype(z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_m3_n4),
            decltype(z_shuffle_thread_desc_m0_n0_m1_n1_m2_n2_n3_m3_n4),
            Sequence<m0, I1, m1, n1, m2, n2, n3, n4, I1>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>,
            8,
            1,
            1,
            true>{z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_m3_n4,
                  make_multi_index(0,           // MRepeat
                                   0,           // NRepeat
                                   wave_id[I0], // MWaveId
                                   wave_id[I1], // NWaveId
                                   wave_m_n_id[I1] / ZN4,
                                   0,
                                   wave_m_n_id[I0],
                                   0,
                                   wave_m_n_id[I1] % ZN4)};

        auto z_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
            ushort,
            ZDataType,
            decltype(z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5),
            decltype(z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5),
            tensor_operation::element_wise::PassThrough,
            Sequence<I1, // MBlockId
                     I1, // NBlockID
                     m0, // MRepeat
                     I1, // NRepeat
                     m1, // MWaveId
                     n1, // NWaveId
                     m2, // MPerXdl
                     n2, // NGroupNum
                     n3, // NInputNum
                     n4>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>,
            9, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>{z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                  make_multi_index(block_work_idx_m, // MBlockId
                                   0,                // NBlockId
                                   0,                // mrepeat
                                   0,                // nrepeat
                                   wave_id[I0],      // MWaveId
                                   wave_id[I1],      // NWaveId
                                   wave_m_n_id[I1],  // MPerXdl
                                   0,                // group
                                   wave_m_n_id[I0],  // NInputIndex
                                   0),
                  tensor_operation::element_wise::PassThrough{}};

        if constexpr(Deterministic)
        {
            block_sync_lds();
        }

	q_blockwise_copy.RunRead(a_grid_desc_ak0_m_ak1,a_grid_buf);
        //Prefetch K[NxKperStep]
        k_blockwise_copy.RunRead(b_grid_desc_bk0_n_bk1,b_grid_buf);
        q_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_ak0_m_ak1,
                                            q_block_slice_copy_step);
        k_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_bk0_n_bk1,
                                            k_block_slice_copy_step);

	// causal mask generation
	// two accumulators
	// upper traingular masked accumulator (-infinity)  when k_block_start == q_block_start
	// zero-initialized accmulator when k_block_start + N_per_block < q_block_start
	//
        // 8d thread_desc in thread scope
        constexpr auto c_thread_lengths =
            qk_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

        // 8d block_desc in block scope
        constexpr auto c_block_lengths =
            qk_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

        constexpr auto M0_1 = c_block_lengths[I0];
        constexpr auto N0_1 = c_block_lengths[I1];
        constexpr auto M1_1 = c_block_lengths[I2];
        constexpr auto N1_1 = c_block_lengths[I3];
        constexpr auto M2_1 = c_block_lengths[I4];
        constexpr auto N2_1 = c_block_lengths[I5];
        constexpr auto N3_1 = c_block_lengths[I6];
        constexpr auto N4_1 = c_block_lengths[I7];

        using Acc0TileIterator = SpaceFillingCurve<
            decltype(c_thread_lengths),
            typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
            typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
            false>; // SnakeCurved
	    
        StaticBuffer<AddressSpaceEnum::Vgpr, FloatGemmAcc, acc_thread_buf.Size(), true>
           masked_acc_thread_buf;


        constexpr auto block_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(M0_1, M1_1, M2_1)),
                           make_unmerge_transform(make_tuple(N0_1, N1_1, N2_1, N3_1, N4_1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));
	auto n_block_idx_on_grid  =
                __builtin_amdgcn_readfirstlane(block_work_idx_m* MPerBlock);
        static_for<0, Acc0TileIterator::GetNumOfAccess(), 1>{}([&](auto idx) {
            auto acc0_idx = Acc0TileIterator::GetIndex(idx) + acc0_thread_origin;
            const index_t m_i =
               block_to_m_n_adaptor.CalculateBottomIndex(acc0_idx)[I0];
            const index_t n_j =
               block_to_m_n_adaptor.CalculateBottomIndex(acc0_idx)[I1];
            auto m_idx = m_i + m_block_data_idx_on_grid;
            auto n_idx = n_j + n_block_idx_on_grid;
            if(c0_matrix_mask.IsMaskedElement(m_idx, n_idx))
            {
                masked_acc_thread_buf(idx) = -ck::NumericLimits<float>::Infinity();
            }
	    else
	    {
                masked_acc_thread_buf(idx) = 0.0f;
	    }
         });
          
        do
        {
	   // find alternate approach for skipping computation for white tile(s)
            auto n_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(gemm1_k_block_outer_index * NPerBlock);
            if(c0_matrix_mask.IsTileSkippable(
                   m_block_data_idx_on_grid, n_block_data_idx_on_grid, MPerBlock, NPerBlock))
            {
                continue;
            }
            // gemm0
	    //
	    //
	    //
            raise_priority();
            {

                //Initialize QK.acc buffer;;
                //FIXME use DS_READ QK.acc, offset(ffffffff),
                //
                //wait for K preload to complete
		//find out why acc initialization didnt happen before ds_write
                if ((MaskOutUpperTriangle || PadN) && ((k_block_start + NPerBlock) >= q_block_start + MPerBlock))
	        {
                   static_for<0, Acc0TileIterator::GetNumOfAccess(), 1>{}([&](auto i) {
                        //acc_element_op(acc_thread_buf(i), masked_acc_thread_buf[i]);
			acc_thread_buf(i) = masked_acc_thread_buf(i);
                   });
		}
	        else
		{
                   acc_thread_buf.Clear();
	        }	
                vm_waitcnt(0);
                q_blockwise_copy.RunWrite(a_block_desc_ak0_m_ak1,q_block_buf);
                k_blockwise_copy.RunWrite(b_block_desc_bk0_n_bk1,k_block_buf);
	        q_blockwise_copy.RunRead(a_grid_desc_ak0_m_ak1,a_grid_buf);
                k_blockwise_copy.RunRead(b_grid_desc_bk0_n_bk1,b_grid_buf);
		//if (num_k_block_main_loop > 1)
		//{
		//}
                //main body
                if (num_k_block_main_loop > 2)
                {
                    index_t i =0;
                    do {

                        lds_waitcnt(0); // wait for LDS write to complete by all WG threads()
                        wg_sync();

                        qk_blockwise_gemm.Run(k_block_buf, q_block_buf, acc_thread_buf);
                        q_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_ak0_m_ak1,
                                            q_block_slice_copy_step);
                        k_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_bk0_n_bk1,
                                            k_block_slice_copy_step);
                        //write i+1 buffer
                        vm_waitcnt(0); // wait for global reads return data
                        q_blockwise_copy.RunWrite(a_block_desc_ak0_m_ak1,q_block_buf);
                        k_blockwise_copy.RunWrite(b_block_desc_bk0_n_bk1,k_block_buf);
                        q_blockwise_copy.RunRead(a_grid_desc_ak0_m_ak1,a_grid_buf);
                        k_blockwise_copy.RunRead(b_grid_desc_bk0_n_bk1,b_grid_buf);
                        i++;

                    } while(i < (num_k_block_main_loop-2));
                }
                {
                //tail
                    lds_waitcnt(0); // wait for LDS write to complete by all WG threads()
                    wg_sync();
                    qk_blockwise_gemm.Run(k_block_buf, q_block_buf, acc_thread_buf);
                    vm_waitcnt(0); // wait for global reads return data
                    q_blockwise_copy.RunWrite(a_block_desc_ak0_m_ak1,q_block_buf);
                    k_blockwise_copy.RunWrite(b_block_desc_bk0_n_bk1,k_block_buf);
                    lds_waitcnt(0); // wait for LDS write to complete by all WG threads()
                    wg_sync();
		    qk_blockwise_gemm.Run(k_block_buf, q_block_buf, acc_thread_buf);
                }


	    }
            lower_priority();

            // 8d thread_desc in thread scope
            //constexpr auto c_thread_lengths =
            //    qk_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

            // 8d block_desc in block scope
            //constexpr auto c_block_lengths =
            //    qk_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

            constexpr auto M0 = c_block_lengths[I0];
            constexpr auto N0 = c_block_lengths[I1];
            constexpr auto M1 = c_block_lengths[I2];
            constexpr auto N1 = c_block_lengths[I3];
            constexpr auto M2 = c_block_lengths[I4];
            constexpr auto N2 = c_block_lengths[I5];
            constexpr auto N3 = c_block_lengths[I6];
            constexpr auto N4 = c_block_lengths[I7];

            // works like multi-dimension static_for (static_ford), but provides both the linear
            // index as well as n-d index
            //using Acc0TileIterator = SpaceFillingCurve<
            //    decltype(c_thread_lengths),
            //    typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
            //    typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
            //    false>; // SnakeCurved

            constexpr auto block_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(M0, M1, M2)),
                           make_unmerge_transform(make_tuple(N0, N1, N2, N3, N4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));

            // do MNK padding or upper triangular masking
            //if ((MaskOutUpperTriangle || PadN) && ((k_block_start + NPerBlock) >= q_block_start + MPerBlock))
	    //if(0)
            //{

            //    static_for<0, Acc0TileIterator::GetNumOfAccess(), 1>{}([&](auto i) {
            //        auto acc0_thread_idx = Acc0TileIterator::GetIndex(i) + acc0_thread_origin;
            //        auto m_local =
            //            block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I0];
            //        auto n_local =
            //            block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I1];
            //        auto m_global = m_local + m_block_data_idx_on_grid;
            //        auto n_global = n_local + n_block_data_idx_on_grid;
            //        if(c0_matrix_mask.IsMaskedElement(m_global, n_global))
            //        {
            //            acc_thread_buf(i) = -ck::NumericLimits<float>::Infinity();
            //        }
            //        else
            //        {
            //            acc_element_op(acc_thread_buf(i), acc_thread_buf[i]);
            //        }
            //    });
            //}
            //else
            //{
            //    static_for<0, acc_thread_buf.Size(), 1>{}(
            //        [&](auto i) { acc_element_op(acc_thread_buf(i), acc_thread_buf[i]); });
            //}
            static_for<0, acc_thread_buf.Size(), 1>{}(
                    [&](auto i) { acc_element_op(acc_thread_buf(i), acc_thread_buf[i]); });

            block_sync_lds(); // wait for lds read in gemm0 blockwise gemm

            // preload gemm1 B data into VGPR 
            b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

            b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                                                 b1_block_slice_copy_step);
            // softmax
            SoftmaxBuf& max = blockwise_softmax.max_value_buf;
            SoftmaxBuf& sum = blockwise_softmax.sum_value_buf;

            blockwise_softmax.Run(acc_thread_buf, workspace_buf);

            constexpr auto position_offset = N3 * N4;
            constexpr auto iterator_offset = n2 * n3 * n4;

            if constexpr(IsDropout) // dropout
            {
                static_for<0, Acc0TileIterator::GetNumOfAccess(), iterator_offset>{}([&](auto i) {
                    auto acc0_thread_idx = Acc0TileIterator::GetIndex(i) + acc0_thread_origin;
                    auto m_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I0];
                    auto n_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I1];
                    auto m_global = m_local + m_block_data_idx_on_grid;
                    auto n_global = n_local + n_block_data_idx_on_grid;

                    auto global_elem_id = z_random_matrix_offset + m_global * raw_n_padded +
                                          n_global; // unique element global 1d id

                    blockwise_dropout.template GenerateZMatrixAttnFwd<decltype(z_tensor_buffer),
                                                                      decltype(n0),
                                                                      decltype(position_offset)>(
                        ph, global_elem_id, z_tensor_buffer);

                    z_tmp_thread_copy_vgpr_to_lds.Run(z_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                      make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                                      z_tensor_buffer,
                                                      z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                      z_block_buf);

                    z_shuffle_thread_copy_lds_to_vgpr.Run(
                        z_shuffle_block_desc_m0_n0_m1_n1_m2_n2_n3_m3_n4,
                        z_block_buf,
                        z_shuffle_thread_desc_m0_n0_m1_n1_m2_n2_n3_m3_n4,
                        make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                        z_tensor_buffer);

                    blockwise_dropout.template ApplyDropoutWithZ<decltype(acc_thread_buf),
                                                                 decltype(z_tensor_buffer),
                                                                 decltype(n0),
                                                                 decltype(i)>(acc_thread_buf,
                                                                              z_tensor_buffer);

                    // save z to global
                    if(p_z_grid)
                    {
                        z_thread_copy_vgpr_to_global.Run(
                            z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                            make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                            z_tensor_buffer,
                            z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                            z_grid_buf);
                        z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                            z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                            make_multi_index(0, 0, 0, 1, 0, 0, 0, 0, 0, 0));
                    }
                });
                z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                    z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                    make_multi_index(0, 0, 0, -(n0.value), 0, 0, 0, 0, 0, 0));
                z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                    z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                    make_multi_index(0, 1, 0, 0, 0, 0, 0, 0, 0, 0));
            }


            raise_priority();
            // gemm1
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // Initialize acc1
                acc1_thread_buf.Clear();

                // preload data into LDS
                //b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

                //b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                //                                     b1_block_slice_copy_step);

                block_sync_lds(); // wait for reduction LDS read

                b1_blockwise_copy.RunWrite(b1_block_desc_bk0_n_bk1, b1_block_buf);

                // main body
                if constexpr(num_gemm1_k_block_inner_loop > 1)
                {
                    static_for<0, num_gemm1_k_block_inner_loop - 1, 1>{}([&](auto i) {
                        a1_blockwise_copy.Run(acc_thread_desc_k0_m_k1,
                                              make_tuple(Number<i * A1ThreadSliceK0>{}, I0, I0),
                                              acc_thread_buf,
                                              a1_thread_desc_k0_m_k1,
                                              make_tuple(I0, I0, I0),
                                              a1_thread_buf);

                        b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

                        block_sync_lds();

                        gemm1_blockwise_gemm.Run(a1_thread_buf, b1_block_buf, acc1_thread_buf);

                        block_sync_lds();

                        b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                                                             b1_block_slice_copy_step);

                        b1_blockwise_copy.RunWrite(b1_block_desc_bk0_n_bk1, b1_block_buf);
                    });
                }
                // tail
                {
                    q_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_ak0_m_ak1,
                                                q_block_reset_copy_step); // rewind K
                    k_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_bk0_n_bk1,
                                                k_block_reset_copy_step); // rewind K and step N
	            q_blockwise_copy.RunRead(a_grid_desc_ak0_m_ak1,a_grid_buf);
                    //Prefetch K[NxKperStep]
                    k_blockwise_copy.RunRead(b_grid_desc_bk0_n_bk1,b_grid_buf);
                    a1_blockwise_copy.Run(
                        acc_thread_desc_k0_m_k1,
                        make_tuple(
                            Number<(num_gemm1_k_block_inner_loop - 1) * A1ThreadSliceK0>{}, I0, I0),
                        acc_thread_buf,
                        a1_thread_desc_k0_m_k1,
                        make_tuple(I0, I0, I0),
                        a1_thread_buf);

                    lds_waitcnt(0); // wait for LDS write to complete by all WG threads()
		    wg_sync();

                    gemm1_blockwise_gemm.Run(a1_thread_buf, b1_block_buf, acc1_thread_buf);
                }
            } // end gemm1
            lower_priority();

            // workaround compiler issue; see ck/ck.hpp
            if constexpr(CK_WORKAROUND_SWDEV_XXXXXX_BF16_ATTEN_FWD_GFX908_ISSUE == 1 &&
                         (is_same_v<FloatGemm, bhalf_t>)&&MPerBlock == 256 && NPerBlock == 128 &&
                         Gemm1NPerBlock == 128)
            {
                __builtin_amdgcn_sched_barrier(0);
            }

            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                gemm1_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();
            constexpr auto cm0 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
            constexpr auto cn0 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
            constexpr auto cm1 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
            constexpr auto cn1 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
            constexpr auto cm2 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
            constexpr auto cn2 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
            constexpr auto cn3 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
            constexpr auto cn4 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);
            constexpr auto c_thread_slice_desc_m_n = make_naive_tensor_descriptor_packed(
                make_tuple(cm0 * cm1 * cm2, cn0 * cn1 * cn2 * cn3 * cn4));
            constexpr auto c_thread_buf_slice_m = c_thread_slice_desc_m_n.GetLength(I0);
            constexpr auto c_thread_buf_slice_n = c_thread_slice_desc_m_n.GetLength(I1);

            // TODO: may convert to log domain
            running_max_new = mathext::max(max, running_max);
            running_sum_new = mathext::exp(running_max - running_max_new) * running_sum +
                              mathext::exp(max - running_max_new) * sum;

            static_for<0, c_thread_buf_slice_m, 1>{}([&](auto iM) {
	            FloatGemmAcc maxAdjust = math::exp(running_max[iM] - running_max_new[iM]);
                    FloatGemmAcc div_scale = 1/running_sum_new[iM];
	            FloatGemmAcc maxAdjust1 = math::exp(max[iM] - running_max_new[iM]);
                static_for<0, c_thread_buf_slice_n, 1>{}([&](auto iN) {
                    auto I = Number<c_thread_slice_desc_m_n.CalculateOffset(make_tuple(iM, iN))>{};
                    FloatGemmAcc acc1 = acc1_thread_buf[I]; // P*V
                    FloatGemmAcc c    = c_thread_buf[I];    // O
                    FloatGemmAcc c_new = running_sum[iM] * maxAdjust * c + maxAdjust1*acc1 /  div_scale;
                        //(running_sum[iM] * math::exp(running_max[iM] - running_max_new[iM]) * c +
                        // math::exp(max[iM] - running_max_new[iM]) * acc1) /
                        //running_sum_new[iM]; // Formula by Dao et al.,
                                             // https://arxiv.org/pdf/2205.14135v2.pdf section 3.1

                    c_thread_buf(I) = c_new; // O_new
                });
            });


            // update before next j iteration
            running_max = running_max_new;
            running_sum = running_sum_new;

            q_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_ak0_m_ak1,
                                            q_block_slice_copy_step);
            k_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_bk0_n_bk1,
                                            k_block_slice_copy_step);
            block_sync_lds(); // wait for gemm1 LDS read
	    k_block_start = k_block_start + NPerBlock;
        } while(++gemm1_k_block_outer_index < num_gemm1_k_block_outer_loop); // end j loop

        // Calculate max + ln(sum) and write out

        if constexpr(IsLseStoring)
        {
            static_for<0, MXdlPerWave, 1>{}(
                [&](auto I) { lse_thread_buf(I) = running_max(I) + math::log(running_sum(I)); });

            if(get_warp_local_1d_id() < AccM2)
            {
                static_for<0, MXdlPerWave, 1>{}([&](auto I) {
                    // copy from VGPR to Global
                    lse_thread_copy_vgpr_to_global.Run(lse_thread_desc_mblock_mrepeat_mwave_mperxdl,
                                                       make_tuple(I0, Number<I>{}, I0, I0),
                                                       lse_thread_buf,
                                                       lse_grid_desc_mblock_mrepeat_mwave_mperxdl,
                                                       lse_grid_buf);

                    lse_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                        lse_grid_desc_mblock_mrepeat_mwave_mperxdl, make_multi_index(0, 1, 0, 0));
                });
            }
        }

        // shuffle C and write out
        {
            static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              Gemm1NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
            constexpr index_t NWave = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);

            // TODO: hacky, fix it!
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                gemm1_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp is only used to get lengths
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
                gemm1_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I4);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I5);
            constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I6);
            constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I7);

            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<FloatCShuffle*>(p_shared),
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                        M1,                                      // M1 = MWave
                        M2)),                                    // M2 = MPerXdl
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                        N1,                                      // N1 = NWave
                        N2,                                      // N2 * N3 * N4 = NPerXdl
                        N3,
                        N4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4>{}, Sequence<>{}, Sequence<1, 3, 5, 6, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                gemm1_blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2, N3, N4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatGemmAcc,
                                                   FloatCShuffle,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   tensor_operation::element_wise::PassThrough,
                                                   Sequence<CShuffleMXdlPerWavePerShuffle,
                                                            CShuffleNXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            I1,
                                                            N2,
                                                            I1,
                                                            N4>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I3],
                                     n_thread_data_on_block_idx[I4]),
                    tensor_operation::element_wise::PassThrough{}};

            // shuffle: blockwise copy C from LDS to global
            auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                         1,
                         CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                FloatCShuffle,        // typename SrcData,
                FloatC,               // typename DstData,
                decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(block_work_idx_m, 0, block_work_idx[I1], 0),
                 c_element_op};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<MXdlPerWave, Gemm1NXdlPerWave, 1, 1, 1, N2, 1, N4>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                  Sequence<CShuffleMXdlPerWavePerShuffle,
                                           CShuffleNXdlPerWavePerShuffle,
                                           1,
                                           1,
                                           1,
                                           N2,
                                           1,
                                           N4>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_c_global =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, Gemm1NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                c_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                              sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                              c_thread_buf,
                                              c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                              c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                c_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    c_shuffle_block_buf,
                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                    c_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
                }
            });
        }
    }
};

} // namespace ck
