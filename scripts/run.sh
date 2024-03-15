#conda activate myigwn-py39
#export OMP_NUM_THREADS=4

OUTPUT_DIR="trained_models/BNS50102432_SVDRealImag_Conv1D_8M_17D"

rm -rf $OUTPUT_DIR
python make_config_ri50.py $OUTPUT_DIR
nohup python train_ri50.py $OUTPUT_DIR >out/nh_SVDRI50_Conv1D_8M_17D.out & 