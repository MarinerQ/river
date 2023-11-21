#conda activate myigwn-py39
#export OMP_NUM_THREADS=24

#OUTPUT_DIR="trained_models/glasnsf_mlpconv2d"
#OUTPUT_DIR="trained_models/glasnsf_mlpconv1d"
OUTPUT_DIR="trained_models/glasnsf_conv1dconv2d"
#OUTPUT_DIR="trained_models/glasnsf_conv1dconv2d_a8d200"
#OUTPUT_DIR="trained_models/test"
#rm -rf $OUTPUT_DIR
#python make_config.py $OUTPUT_DIR
nohup python train.py $OUTPUT_DIR >out/nh_2.out & 