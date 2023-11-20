#conda activate myigwn-py39
#export OMP_NUM_THREADS=24

OUTPUT_DIR="trained_models/glasnsf_mlpconv2d"
#OUTPUT_DIR="trained_models/test"
python make_config.py $OUTPUT_DIR
nohup python train.py $OUTPUT_DIR >out/nh1.out & 