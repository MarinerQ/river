# conda activate myigwn-py39 && export OMP_NUM_THREADS=4

#OUTPUT_DIR="trained_models/BNS50102432_ConvRealImag_Conv1D_8M_17D_5"

#rm -rf $OUTPUT_DIR
#python make_config_ri50.py $OUTPUT_DIR
#nohup python train_ri50.py $OUTPUT_DIR >out/nh_ConvRI50_Conv1D_8M_17D_5.out & 



OUTPUT_DIR="trained_models/BNS20MB_8M"

rm -rf $OUTPUT_DIR
python make_config_mb.py $OUTPUT_DIR
nohup python train_mb.py $OUTPUT_DIR >out/nh_mb8M.out & 