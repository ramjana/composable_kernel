#ifndef CK_BLOCKWISE_TENSOR_SLICE_TRANSFER_V6R2_HPP
#define CK_BLOCKWISE_TENSOR_SLICE_TRANSFER_V6R2_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "cluster_descriptor.hpp"
#include "threadwise_tensor_slice_transfer_v6r2.hpp"

namespace ck {

// this version does following things to avoid scratch memory issue
// 1. Use StaticallyIndexedArray instead of C array for thread buffer
// 2. It does not keep reference to tensor descriptor
// 3. Run() does not construct new tensor coordinate
template <index_t BlockSize,
          typename ElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename BlockSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename Src0Data,
          typename Src1Data,
          typename DstData,
          typename Src0Desc,
          typename Src1Desc,
          typename DstDesc,
          typename DimAccessOrder,
          index_t VectorDim,
          index_t ScalarPerVector,
          bool ThreadTransferSrc0ResetCoordinateAfterRun,
          bool ThreadTransferSrc1ResetCoordinateAfterRun,
          bool ThreadTransferDstResetCoordinateAfterRun>
struct BlockwiseTensorSliceTransfer_v6r2
{
    static constexpr index_t nDim = remove_reference_t<Src0Desc>::GetNumOfDimension();

    static constexpr auto thread_slice_lengths = BlockSliceLengths{} / ThreadClusterLengths{};

    using Index = MultiIndex<nDim>;

    __device__ constexpr BlockwiseTensorSliceTransfer_v6r2(const Src0Desc& src0_desc,
                                                           const Index& src0_block_slice_origin,
                                                           const Src1Desc& src1_desc,
                                                           const Index& src1_block_slice_origin,
                                                           const DstDesc& dst_desc,
                                                           const Index& dst_block_slice_origin,
                                                           const ElementwiseOperation& element_op)
        : threadwise_transfer_(src0_desc,
                               make_zero_multi_index<nDim>(),
                               src1_desc,
                               make_zero_multi_index<nDim>(),
                               dst_desc,
                               make_zero_multi_index<nDim>(),
                               element_op)

    {
        static_assert(nDim == remove_reference_t<remove_cv_t<Src0Desc>>::GetNumOfDimension() &&
                          nDim == remove_reference_t<remove_cv_t<Src1Desc>>::GetNumOfDimension() &&
                          nDim == remove_reference_t<remove_cv_t<DstDesc>>::GetNumOfDimension() &&
                          nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == DimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(thread_slice_lengths * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(BlockSize >= thread_cluster_desc_.GetElementSize(),
                      "wrong! BlockSize too small");

        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
                make_multi_index(get_thread_local_1d_id()));

            const auto thread_data_idx_begin = thread_cluster_idx * thread_slice_lengths;

            threadwise_transfer_.SetSrc0SliceOrigin(
                src0_desc, src0_block_slice_origin + thread_data_idx_begin);
            threadwise_transfer_.SetSrc1SliceOrigin(
                src1_desc, src1_block_slice_origin + thread_data_idx_begin);
            threadwise_transfer_.SetDstSliceOrigin(dst_desc,
                                                   dst_block_slice_origin + thread_data_idx_begin);
        }
    }

    template <typename Src0Buffer, typename Src1Buffer, typename DstBuffer>
    __device__ void Run(const Src0Desc& src0_desc,
                        const Src0Buffer& src0_buf,
                        const Src1Desc& src1_desc,
                        const Src1Buffer& src1_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.Run(src0_desc, src0_buf, src1_desc, src1_buf, dst_desc, dst_buf);
        }
    }

    __device__ void MoveSrc0SliceWindow(const Src0Desc& src0_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrc0SliceWindow(src0_desc, step);
        }
    }

    __device__ void MoveSrc1SliceWindow(const Src1Desc& src1_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrc1SliceWindow(src1_desc, step);
        }
    }

    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveDstSliceWindow(dst_desc, step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    using ThreadwiseTransfer =
        ThreadwiseTensorSliceTransfer_v6r2<Src0Data,
                                           Src1Data,
                                           DstData,
                                           Src0Desc,
                                           Src1Desc,
                                           DstDesc,
                                           ElementwiseOperation,
                                           decltype(thread_slice_lengths),
                                           DimAccessOrder,
                                           VectorDim,
                                           ScalarPerVector,
                                           DstInMemOp,
                                           ThreadTransferSrc0ResetCoordinateAfterRun,
                                           ThreadTransferSrc1ResetCoordinateAfterRun,
                                           ThreadTransferDstResetCoordinateAfterRun>;

    ThreadwiseTransfer threadwise_transfer_;
};

} // namespace ck
#endif
