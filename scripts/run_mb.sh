# conda activate myigwn-py39 && export OMP_NUM_THREADS=4

OUTPUT_DIR="trained_models/BNS20MB_8M_12D_3"

rm -rf $OUTPUT_DIR
python make_config_mb.py $OUTPUT_DIR
nohup python train_mb.py $OUTPUT_DIR >out/nh_mb8M_12d_3.out & 