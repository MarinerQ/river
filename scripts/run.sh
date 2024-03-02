#conda activate myigwn-py39
#export OMP_NUM_THREADS=10

OUTPUT_DIR="trained_models/BNS50102432_RealImag_Conv1D_1M"

rm -rf $OUTPUT_DIR
python make_config_ri50.py $OUTPUT_DIR
nohup python train_ri50.py $OUTPUT_DIR >out/nh_RI50_Conv1D_1M.out & 