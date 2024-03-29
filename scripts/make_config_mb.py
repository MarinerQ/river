import json
import os
import numpy as np
import sys
from river.data.utils import PARAMETER_NAMES_CONTEXT_PRECESSINGBNS_BILBY, PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
from river.models.embedding.conv import EmbeddingConv1D, EmbeddingConv2D

if __name__ == "__main__":
    config_path = sys.argv[1]

    config_dict = {}
    #ckpt_dir = 'trained_models/glasnsf_mlpconv2d'
    ckpt_dir = config_path
    config_dict['ckpt_dir'] = ckpt_dir
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
        print(f"Made dir {ckpt_dir}")
        
    # data_generator_parameters
    config_dict['data_generator_parameters'] = {}
    config_dict['data_generator_parameters']['source_type'] = 'BNS'
    config_dict['data_generator_parameters']['detector_names'] = ['H1', 'L1', 'V1']
    config_dict['data_generator_parameters']['full_duration'] = 320
    config_dict['data_generator_parameters']['f_low'] = 20
    config_dict['data_generator_parameters']['f_ref'] = 20
    config_dict['data_generator_parameters']['parameter_names'] = PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
    config_dict['data_generator_parameters']['context_parameter_names'] = PARAMETER_NAMES_CONTEXT_PRECESSINGBNS_BILBY
    config_dict['data_generator_parameters']['use_sealgw_detector'] = True
    config_dict['data_generator_parameters']['snr_threshold'] = 8
    config_dict['data_generator_parameters']['d_min'] = 10
    config_dict['data_generator_parameters']['d_max'] = 100
    config_dict['data_generator_parameters']['d_power'] = 1
    config_dict['data_generator_parameters']['tc_min'] = -0.1
    config_dict['data_generator_parameters']['tc_max'] = 0.1
    config_dict['data_generator_parameters']['timing_std'] = 0.01

    # pre-calculated data parameters
    config_dict['precaldata_parameters'] = {}
    config_dict['precaldata_parameters']['train'] = {}
    config_dict['precaldata_parameters']['train']['folder'] = '/home/qian.hu/mlpe/training_data/bns_20Hz_mb/train'

    config_dict['precaldata_parameters']['valid'] = {}
    config_dict['precaldata_parameters']['valid']['folder'] = '/home/qian.hu/mlpe/training_data/bns_20Hz_mb/valid'

    config_dict['precaldata_parameters']['asd_path'] = '/home/qian.hu/mlpe/training_data/psd/LHVdesign'



    # training_parameters
    config_dict['training_parameters'] = {}
    config_dict['training_parameters']['batch_size_train'] = 8192
    config_dict['training_parameters']['minibatch_size_train'] = 1024
    config_dict['training_parameters']['batch_size_valid'] = 500
    config_dict['training_parameters']['minibatch_size_valid'] = 1
    config_dict['training_parameters']['batch_size_test'] = 10
    config_dict['training_parameters']['lr'] = 5e-4
    config_dict['training_parameters']['weight_decay'] = 1e-5
    config_dict['training_parameters']['gamma'] = 0.7
    config_dict['training_parameters']['max_epoch'] = 1000
    config_dict['training_parameters']['epoches_pretrain'] = 10
    config_dict['training_parameters']['epoches_save_loss'] = 5
    config_dict['training_parameters']['epoches_adjust_lr'] = 8
    config_dict['training_parameters']['epoches_adjust_lr_again'] = 6
    config_dict['training_parameters']['load_from_previous_train'] = False

    config_dict['training_parameters']['device'] = 'cuda:0'



    # model_parameters
    config_dict['model_parameters'] = {}

    # Embedding
    config_dict['model_parameters']['embedding'] = {}
    config_dict['model_parameters']['embedding']['model'] = 'SimpleViT'
    NCOND = 256
    config_dict['model_parameters']['embedding']['seq_len'] = 3328
    config_dict['model_parameters']['embedding']['patch_size'] = 256
    config_dict['model_parameters']['embedding']['num_classes'] = NCOND
    config_dict['model_parameters']['embedding']['dim'] = 1024 # 1024
    config_dict['model_parameters']['embedding']['depth'] = 6 # 6
    config_dict['model_parameters']['embedding']['heads'] = 8 # 8
    config_dict['model_parameters']['embedding']['mlp_dim'] = 2048 # 2048
    config_dict['model_parameters']['embedding']['channels'] = 6 # (real+imag) * Ndet
    config_dict['model_parameters']['embedding']['dim_head'] = 64 # 64

    
    # flow
    config_dict['model_parameters']['flow'] = {}

    config_dict['model_parameters']['flow']['model'] = 'CouplingNSF'
    config_dict['model_parameters']['flow']['n_inputs'] = 17
    config_dict['model_parameters']['flow']['n_transforms'] = 15
    config_dict['model_parameters']['flow']['n_conditional_inputs'] = NCOND
    config_dict['model_parameters']['flow']['n_neurons'] = 96  # 32 by default
    config_dict['model_parameters']['flow']['batch_norm_between_transforms'] = True
    config_dict['model_parameters']['flow']['batch_norm_within_blocks'] = True
    config_dict['model_parameters']['flow']['n_blocks_per_transform'] = 3  # 2 by default, 5
    config_dict['model_parameters']['flow']['num_bins'] = 6  # 4 by default, 8
    config_dict['model_parameters']['flow']['tail_bound'] = 1 # 5 by default, 1




    with open(f"{ckpt_dir}/config.json", 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
