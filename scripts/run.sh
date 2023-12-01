#conda activate myigwn-py39
#export OMP_NUM_THREADS=10

OUTPUT_DIR="trained_models/GlasNSFConv1DRes_100ktest"
#OUTPUT_DIR="trained_models/test"
#OUTPUT_DIR="trained_models/zukonsf_conv1dconv2d_a1d200"
rm -rf $OUTPUT_DIR
python make_config3.py $OUTPUT_DIR
nohup python train3.py $OUTPUT_DIR >out/nh_1.out & 