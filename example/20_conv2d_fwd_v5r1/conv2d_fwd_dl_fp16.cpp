#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "tensor_layout.hpp"
#include "device_conv2d_fwd_v5r1_dl_nchwc_kcyxc_nkhwk.hpp"
#include "element_wise_operation.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// using InLayout  = ck::tensor_layout::convolution::NHWC;
// using WeiLayout = ck::tensor_layout::convolution::KYXC;
// using OutLayout = ck::tensor_layout::convolution::NHWK;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

constexpr ck::index_t CK_CONFIG_N  = 1;
constexpr ck::index_t CK_CONFIG_K0 = 4;
constexpr ck::index_t CK_CONFIG_K1 = 8;
constexpr ck::index_t CK_CONFIG_C0 = 4;
constexpr ck::index_t CK_CONFIG_C1 = 8;

constexpr ck::index_t CK_CONFIG_Y = 3;
constexpr ck::index_t CK_CONFIG_X = 3;

constexpr ck::index_t CK_CONFIG_HI = 1080;
constexpr ck::index_t CK_CONFIG_WI = 1920;

constexpr ck::index_t CK_CONFIG_CONV_STRIDE_H = 1;
constexpr ck::index_t CK_CONFIG_CONV_STRIDE_W = 1;

constexpr ck::index_t CK_CONFIG_CONV_DILATION_H = 1;
constexpr ck::index_t CK_CONFIG_CONV_DILATION_W = 1;

constexpr ck::index_t CK_CONFIG_IN_LEFT_PAD_H = 1;
constexpr ck::index_t CK_CONFIG_IN_LEFT_PAD_W = 1;

constexpr ck::index_t CK_CONFIG_IN_RIGHT_PAD_H = 1;
constexpr ck::index_t CK_CONFIG_IN_RIGHT_PAD_W = 1;

constexpr ck::index_t CK_CONFIG_BLOCK_SIZE = 256;

constexpr ck::index_t CK_CONFIG_E1 = CK_CONFIG_C0 * CK_CONFIG_Y * CK_CONFIG_X;
constexpr ck::index_t CK_CONFIG_E2 = CK_CONFIG_C1;
constexpr ck::index_t CK_CONFIG_K2 = 4;

constexpr ck::index_t CK_CONFIG_E0_PER_BLOCK = 1;
constexpr ck::index_t CK_CONFIG_E1_PER_BLOCK = 2;

constexpr ck::index_t CK_CONFIG_K_PER_BLOCK  = 16;
constexpr ck::index_t CK_CONFIG_HO_PER_BLOCK = 16;
constexpr ck::index_t CK_CONFIG_WO_PER_BLOCK = 64;

constexpr ck::index_t CK_CONFIG_K_PER_THREAD  = 16;
constexpr ck::index_t CK_CONFIG_HO_PER_THREAD = 2;
constexpr ck::index_t CK_CONFIG_WO_PER_THREAD = 2;
constexpr ck::index_t CK_CONFIG_E_PER_THREAD  = 1;

using CK_CONFIG_ABLOCK_TRANS_THREAD_SLICE_LENGTHS =
    ck::Sequence<1, CK_CONFIG_Y * CK_CONFIG_X, 1, 1, CK_CONFIG_C1>;
using CK_CONFIG_ABLOCK_TRANS_THREAD_CLUSTER_LENGTHS =
    ck::Sequence<1, CK_CONFIG_C0, 1, CK_CONFIG_K_PER_BLOCK, 1>;

constexpr ck::index_t CK_CONFIG_ABLOCK_TRANS_SRC_VEC  = CK_CONFIG_C1;
constexpr ck::index_t CK_CONFIG_ABLOCK_TRANS_DST_VEC  = CK_CONFIG_C1;
constexpr ck::index_t CK_CONFIG_BTHREAD_TRANS_SRC_VEC = CK_CONFIG_C1;
constexpr ck::index_t CK_CONFIG_BTHREAD_TRANS_DST_VEC = CK_CONFIG_K1;

using DeviceConvFwdInstance = ck::tensor_operation::device::
    DeviceConv2dFwdv5r1Xdl_Input_N_C0_Hi_Wi_C1_Weight_K_C0_Y_X_C1_Output_N_K0_Ho_Wo_K1<
        InDataType,
        WeiDataType,
        OutDataType,
        AccDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvFwdDefault,
        CK_CONFIG_BLOCK_SIZE,
        CK_CONFIG_E1,
        CK_CONFIG_E2,
        CK_CONFIG_K2,
        CK_CONFIG_K_PER_BLOCK,
        CK_CONFIG_HO_PER_BLOCK,
        CK_CONFIG_WO_PER_BLOCK,
        CK_CONFIG_E0_PER_BLOCK,
        CK_CONFIG_E1_PER_BLOCK,
        CK_CONFIG_K_PER_THREAD,
        CK_CONFIG_HO_PER_THREAD,
        CK_CONFIG_WO_PER_THREAD,
        CK_CONFIG_E_PER_THREAD,
        CK_CONFIG_ABLOCK_TRANS_THREAD_SLICE_LENGTHS,
        CK_CONFIG_ABLOCK_TRANS_THREAD_CLUSTER_LENGTHS,
        CK_CONFIG_ABLOCK_TRANS_SRC_VEC,
        CK_CONFIG_ABLOCK_TRANS_DST_VEC,
        CK_CONFIG_BTHREAD_TRANS_SRC_VEC,
        CK_CONFIG_BTHREAD_TRANS_DST_VEC>;

template <typename TIn,
          typename TWei,
          typename TOut,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp>
void host_verify(const Tensor<TIn>& in,
                 const Tensor<TWei>& wei,
                 Tensor<TOut>& out,
                 const ConvStrides& conv_strides,
                 const ConvDilations& conv_dilations,
                 const InLeftPads& in_left_pads,
                 const InRightPads&,
                 const InElementOp& in_element_op,
                 const WeiElementOp& wei_element_op,
                 const OutElementOp& out_element_op)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    auto f_nchw = [&](auto n, auto k0, auto ho, auto wo, auto k1) {
        double v = 0;

        const int k = k0 * out.mDesc.GetLengths()[4] + k1;

        for(int c0 = 0; c0 < wei.mDesc.GetLengths()[1]; ++c0)
        {
            for(int y = 0; y < wei.mDesc.GetLengths()[2]; ++y)
            {
                int hi = ho * conv_strides[I0] + y * conv_dilations[I0] - in_left_pads[I0];
                for(int x = 0; x < wei.mDesc.GetLengths()[3]; ++x)
                {
                    int wi = wo * conv_strides[I1] + x * conv_dilations[I1] - in_left_pads[I1];
                    if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                       wi < in.mDesc.GetLengths()[3])
                    {
                        for(int c1 = 0; c1 < wei.mDesc.GetLengths()[4]; ++c1)
                        {
                            // v += in_element_op(static_cast<const double>(in(n, c, hi, wi))) *
                            // wei_element_op(static_cast<const double>(wei(k, c, y, x)));
                            v += static_cast<const double>(in(n, c0, hi, wi, c1)) *
                                 static_cast<const double>(wei(k, c0, y, x, c1));
                        }
                    }
                }
            }
        }
        // double v2 = out(n, k0, ho, wo, k1);

        // out_element_op(v2, v);

        out(n, k0, ho, wo, k1) = v;
    };

    make_ParallelTensorFunctor(f_nchw,
                               out.mDesc.GetLengths()[0],
                               out.mDesc.GetLengths()[1],
                               out.mDesc.GetLengths()[2],
                               out.mDesc.GetLengths()[3],
                               out.mDesc.GetLengths()[4])(std::thread::hardware_concurrency());
}

int main(int argc, char* argv[])
{
    bool do_verification = 0;
    int init_method      = 0;
    int nrepeat          = 5;

    // Conv shape
#if 0
    ck::index_t N               = 128;
    ck::index_t K0              = 256;
    ck::index_t K1              = 256;
    ck::index_t C0              = 192;
    ck::index_t C1              = 192;
    ck::index_t Y               = 3;
    ck::index_t X               = 3;
    ck::index_t Hi              = 71;
    ck::index_t Wi              = 71;
    ck::index_t conv_stride_h   = 2;
    ck::index_t conv_stride_w   = 2;
    ck::index_t conv_dilation_h = 1;
    ck::index_t conv_dilation_w = 1;
    ck::index_t in_left_pad_h   = 1;
    ck::index_t in_left_pad_w   = 1;
    ck::index_t in_right_pad_h  = 1;
    ck::index_t in_right_pad_w  = 1;
#else
    ck::index_t N               = CK_CONFIG_N;
    ck::index_t K0              = CK_CONFIG_K0;
    ck::index_t K1              = CK_CONFIG_K1;
    ck::index_t C0              = CK_CONFIG_C0;
    ck::index_t C1              = CK_CONFIG_C1;
    ck::index_t Y               = CK_CONFIG_Y;
    ck::index_t X               = CK_CONFIG_X;
    ck::index_t Hi              = CK_CONFIG_HI;
    ck::index_t Wi              = CK_CONFIG_WI;
    ck::index_t conv_stride_h   = CK_CONFIG_CONV_STRIDE_H;
    ck::index_t conv_stride_w   = CK_CONFIG_CONV_STRIDE_W;
    ck::index_t conv_dilation_h = CK_CONFIG_CONV_DILATION_H;
    ck::index_t conv_dilation_w = CK_CONFIG_CONV_DILATION_W;
    ck::index_t in_left_pad_h   = CK_CONFIG_IN_LEFT_PAD_H;
    ck::index_t in_left_pad_w   = CK_CONFIG_IN_LEFT_PAD_W;
    ck::index_t in_right_pad_h  = CK_CONFIG_IN_RIGHT_PAD_H;
    ck::index_t in_right_pad_w  = CK_CONFIG_IN_RIGHT_PAD_W;
#endif

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
    }
    else if(argc == 21)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
        N               = std::stoi(argv[4]);
        K0              = std::stoi(argv[5]);
        K1              = std::stoi(argv[6]);
        C0              = std::stoi(argv[7]);
        C1              = std::stoi(argv[8]);
        Y               = std::stoi(argv[9]);
        X               = std::stoi(argv[10]);
        Hi              = std::stoi(argv[11]);
        Wi              = std::stoi(argv[12]);
        conv_stride_h   = std::stoi(argv[13]);
        conv_stride_w   = std::stoi(argv[14]);
        conv_dilation_h = std::stoi(argv[15]);
        conv_dilation_w = std::stoi(argv[16]);
        in_left_pad_h   = std::stoi(argv[17]);
        in_left_pad_w   = std::stoi(argv[18]);
        in_right_pad_h  = std::stoi(argv[19]);
        in_right_pad_w  = std::stoi(argv[20]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        printf(
            "arg4 to 20: N, K0, K1, C0, C1, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
            "RightPx\n");
        exit(0);
    }

    const ck::index_t K = K0 * K1;

    const ck::index_t YEff = (Y - 1) * conv_dilation_h + 1;
    const ck::index_t XEff = (X - 1) * conv_dilation_w + 1;

    const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
    const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;

    const std::vector<ck::index_t> conv_filter_strides{{conv_stride_h, conv_stride_w}};
    const std::vector<ck::index_t> conv_filter_dilations{{conv_dilation_h, conv_dilation_w}};
    const std::vector<ck::index_t> input_left_pads{{in_left_pad_h, in_left_pad_w}};
    const std::vector<ck::index_t> input_right_pads{{in_right_pad_h, in_right_pad_w}};

    // tensor layout
    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C0_, std::size_t H, std::size_t W, std::size_t C1_) {
            {
                return HostTensorDescriptor(
                    std::vector<std::size_t>({N_, C0_, H, W, C1_}),
                    std::vector<std::size_t>({C0_ * H * W * C1_, H * W * C1_, W * C1_, C1_, 1}));
            }
        };

    Tensor<InDataType> in_n_c0_hi_wi_c1(f_host_tensor_descriptor(N, C0, Hi, Wi, C1));
    Tensor<WeiDataType> wei_k_c0_y_x_c1(f_host_tensor_descriptor(K, C0, Y, X, C1));
    Tensor<OutDataType> out_n_k0_ho_wo_k1_host_result(f_host_tensor_descriptor(N, K0, Ho, Wo, K1));
    Tensor<OutDataType> out_n_k0_ho_wo_k1_device_result(
        f_host_tensor_descriptor(N, K0, Ho, Wo, K1));

    std::cout << "in_n_c0_hi_wi_c1: " << in_n_c0_hi_wi_c1.mDesc << std::endl;
    std::cout << "wei_k_c0_y_x_c1: " << wei_k_c0_y_x_c1.mDesc << std::endl;
    std::cout << "out_n_k0_ho_wo_k1: " << out_n_k0_ho_wo_k1_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in_n_c0_hi_wi_c1.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei_k_c0_y_x_c1.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        in_n_c0_hi_wi_c1.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei_k_c0_y_x_c1.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c0_hi_wi_c1.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_c0_y_x_c1.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_k0_ho_wo_k1_device_result.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in_n_c0_hi_wi_c1.mData.data());
    wei_device_buf.ToDevice(wei_k_c0_y_x_c1.mData.data());

    // do GEMM
    auto conv     = DeviceConvFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                      static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                      static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                      N,
                                      K0,
                                      K1,
                                      C0,
                                      C1,
                                      std::vector<ck::index_t>{{Hi, Wi}},
                                      std::vector<ck::index_t>{{Y, X}},
                                      std::vector<ck::index_t>{{Ho, Wo}},
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      InElementOp{},
                                      WeiElementOp{},
                                      OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float ave_time = invoker.Run(argument, nrepeat);

    std::size_t flop = std::size_t(2) * N * K * Ho * Wo * C0 * C1 * Y * X;

    std::size_t num_btype = sizeof(InDataType) * (N * C0 * Hi * Wi * C1) +
                            sizeof(WeiDataType) * (K * C0 * Y * X * C1) +
                            sizeof(OutDataType) * (N * K0 * Ho * Wo * K1);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        host_verify(in_n_c0_hi_wi_c1,
                    wei_k_c0_y_x_c1,
                    out_n_k0_ho_wo_k1_host_result,
                    conv_filter_strides,
                    conv_filter_dilations,
                    input_left_pads,
                    input_right_pads,
                    InElementOp{},
                    WeiElementOp{},
                    OutElementOp{});

        out_device_buf.FromDevice(out_n_k0_ho_wo_k1_device_result.mData.data());

        check_error(out_n_k0_ho_wo_k1_host_result, out_n_k0_ho_wo_k1_device_result);
    }
}
