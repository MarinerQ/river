# conda activate myigwn-py39 && export OMP_NUM_THREADS=4

RUNID=2
RUNLABEL=BNS50102432_SVDConv_

OUTPUT_DIR="trained_models/$RUNLABEL$RUNID"

rm -rf $OUTPUT_DIR
python make_config_ri50.py $OUTPUT_DIR
nohup python train_ri50.py $OUTPUT_DIR >out/nh_$RUNLABEL$RUNID.out & 



#OUTPUT_DIR="trained_models/BNS20MB_8M_17D_4"

#rm -rf $OUTPUT_DIR
#python make_config_mb.py $OUTPUT_DIR
#nohup python train_mb.py $OUTPUT_DIR >out/nh_mb8M_17d.out & 