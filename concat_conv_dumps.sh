
dump_dict=concat_conv1x1_bias_activ_1080p_c8_dumps_b256

conv_fig=../host/driver_offline/include/ck_conv_fig.h

op=resize_concat_conv_bias_activ_fwd_driver_offline_nchwc


echo '' > $conv_fig

n=$1
h=$2
w=$3
y=$4
x=$5
conv1_c0=$6
conv2_c0=$7
c1=$8
k0=$9
k1=${10}
p=${11}
q=${12}
a=${13}

echo "n: $n h: $h w: $w y: $y x: $x conv1_c0: $conv1_c0 conv2_c0: $conv2_c0 c1: $c1 k0: $k0 k1: $k1 pad: {$p, $q} activ: $a"

echo "#define USE_CONV_FIG 1" >> $conv_fig

echo "#define CONV_N $n" >> $conv_fig

echo "#define CONV1_C0_ $conv1_c0" >> $conv_fig
echo "#define CONV1_HI_ $h" >> $conv_fig
echo "#define CONV1_WI_ $w" >> $conv_fig
echo "#define CONV1_C1_ $c1" >> $conv_fig

echo "#define CONV2_C0_ $conv2_c0" >> $conv_fig
echo "#define CONV2_HI_ $h" >> $conv_fig
echo "#define CONV2_WI_ $w" >> $conv_fig
echo "#define CONV2_C1_ $c1" >> $conv_fig

echo "#define CONV_Y $y" >> $conv_fig
echo "#define CONV_X $x" >> $conv_fig
echo "#define CONV_K0 $k0" >> $conv_fig
echo "#define CONV_K1 $k1" >> $conv_fig
echo "#define CONV_STRIDE_H 1" >> $conv_fig
echo "#define CONV_STRIDE_W 1" >> $conv_fig
echo "#define CONV_DILATION_H 1" >> $conv_fig
echo "#define CONV_DILATION_W 1" >> $conv_fig
echo "#define CONV_IN_LEFT_PAD_H $p" >> $conv_fig
echo "#define CONV_IN_LEFT_PAD_W $q" >> $conv_fig
echo "#define CONV_IN_RIGHT_PAD_H $p" >> $conv_fig
echo "#define CONV_IN_RIGHT_PAD_W $q" >> $conv_fig

echo "#define CONV_ACTIV $a" >> $conv_fig

echo "#define CONV_BLOCK_SIZE 256" >> $conv_fig

echo "#define CONV_E1 C0 * Y * X" >> $conv_fig
echo "#define CONV_E2 C1" >> $conv_fig
echo "#define CONV_K2 4" >> $conv_fig

echo "#define CONV_E0_PER_BLOCK 1" >> $conv_fig
echo "#define CONV_K_PER_BLOCK 16" >> $conv_fig
echo "#define CONV_HO_PER_BLOCK 16" >> $conv_fig
echo "#define CONV_WO_PER_BLOCK 64" >> $conv_fig
echo "#define CONV_E1_PER_BLOCK 1" >> $conv_fig

echo "#define CONV_K_PER_THREAD 16" >> $conv_fig
echo "#define CONV_HO_PER_THREAD 2" >> $conv_fig
echo "#define CONV_WO_PER_THREAD 2" >> $conv_fig
echo "#define CONV_E_PER_THREAD 1" >> $conv_fig

echo "#define CONV_ABLOCK_TRANS_THREAD_SLICE_LENGTHS 1, Y * X, 1, 1, C1" >> $conv_fig
echo "#define CONV_ABLOCK_TRANS_THREAD_CLUSTER_LENGTHS 1, C0, 1, KPerBlock, 1" >> $conv_fig 

make -j $op

./host/driver_offline/$op 0 1 4 0 5 2>&1 | tee log

kernel=`sed -n -e '/^input/p' log`
tparm=`sed -n -e '/^BlockSize/p' log`

echo $kernel
echo $tparm

mkdir -p ../$dump_dict/$kernel

rm ../$dump_dict/$kernel/*

cp host/driver_offline/$op-hip-amdgcn-amd-amdhsa-gfx1030.* ../$dump_dict/$kernel

touch ../$dump_dict/$kernel/$tparm

ls -ls ../$dump_dict/$kernel
