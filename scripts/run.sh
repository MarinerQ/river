#conda activate myigwn-py39
#export OMP_NUM_THREADS=10

#OUTPUT_DIR="trained_models/glasnsf_mlpconv2d"
#OUTPUT_DIR="trained_models/glasnsf_mlpconv1d"
#OUTPUT_DIR="trained_models/glasnsf_conv1dconv2d_a1d200_smallbatch"
#OUTPUT_DIR="trained_models/glasnsf_conv1dconv2d_a1d200_ri"
#OUTPUT_DIR="trained_models/glasnsf_conv1dconv2d_a8d200"
OUTPUT_DIR="trained_models/glasnsf_conv1d_a1d200_3"
#OUTPUT_DIR="trained_models/test"
#OUTPUT_DIR="trained_models/zukonsf_conv1dconv2d_a1d200"
rm -rf $OUTPUT_DIR
python make_config.py $OUTPUT_DIR
nohup python train2.py $OUTPUT_DIR >out/nh_3.out & 