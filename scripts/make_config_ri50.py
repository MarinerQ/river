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
    config_dict['data_generator_parameters']['duration'] = 32
    config_dict['data_generator_parameters']['f_low'] = 50
    config_dict['data_generator_parameters']['f_ref'] = 20
    config_dict['data_generator_parameters']['sampling_frequency'] = 2048
    config_dict['data_generator_parameters']['waveform_approximant'] = 'IMRPhenomPv2_NRTidal'
    config_dict['data_generator_parameters']['parameter_names'] = PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
    config_dict['data_generator_parameters']['context_parameter_names'] = PARAMETER_NAMES_CONTEXT_PRECESSINGBNS_BILBY
    config_dict['data_generator_parameters']['PSD_type'] = 'bilby_default' #'bilby_default', zero_noise
    config_dict['data_generator_parameters']['use_sealgw_detector'] = True
    config_dict['data_generator_parameters']['snr_threshold'] = 8
    config_dict['data_generator_parameters']['selection_factor'] = 2
    config_dict['data_generator_parameters']['a_max'] = 0.1
    config_dict['data_generator_parameters']['d_min'] = 10
    config_dict['data_generator_parameters']['d_max'] = 200
    config_dict['data_generator_parameters']['d_power'] = 1
    config_dict['data_generator_parameters']['tc_min'] = -0.1
    config_dict['data_generator_parameters']['tc_max'] = 0.1

    # pre-calculated data parameters
    config_dict['precaldata_parameters'] = {}
    config_dict['precaldata_parameters']['train'] = {}
    config_dict['precaldata_parameters']['train']['folder'] = '/home/qian.hu/mlpe/training_data/bns_50Hz1024Hz32s_lowspin_lb/train'
    #config_dict['precaldata_parameters']['train']['nbatch'] = 10
    #config_dict['precaldata_parameters']['train']['file_per_batch'] = 500
    #config_dict['precaldata_parameters']['train']['sample_per_file'] = 200 
    #config_dict['precaldata_parameters']['train']['filename_prefix'] = 'bns_320s_dataproj_lowspin'

    config_dict['precaldata_parameters']['valid'] = {}
    config_dict['precaldata_parameters']['valid']['folder'] = '/home/qian.hu/mlpe/training_data/bns_50Hz1024Hz32s_lowspin_lb/valid'
    #config_dict['precaldata_parameters']['valid']['nbatch'] = 1
    #config_dict['precaldata_parameters']['valid']['file_per_batch'] = 1
    #config_dict['precaldata_parameters']['valid']['sample_per_file'] = 1000
    #config_dict['precaldata_parameters']['valid']['filename_prefix'] = 'bns_320s_dataproj_lowspin'



    # training_parameters
    config_dict['training_parameters'] = {}
    #config_dict['training_parameters']['Nsample'] = 1000000
    #config_dict['training_parameters']['Nvalid'] = 1000
    config_dict['training_parameters']['batch_size_train'] = 8192
    config_dict['training_parameters']['minibatch_size_train'] = 1024
    config_dict['training_parameters']['batch_size_valid'] = 500
    config_dict['training_parameters']['minibatch_size_valid'] = 100
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

    config_dict['training_parameters']['device'] = 'cuda:1'



    # model_parameters
    config_dict['model_parameters'] = {}
    config_dict['model_parameters']['Vhfile'] = '/home/qian.hu/mlpe/river/test/outputs/Vh_50Hz1024Hz32s.pickle'
    config_dict['model_parameters']['Nbasis'] = 512

    # Embedding - projection
    config_dict['model_parameters']['embedding'] = {}
    #config_dict['model_parameters']['embedding']['model'] = 'EmbeddingMLP1D'
    #config_dict['model_parameters']['embedding']['ndet'] = 3
    #config_dict['model_parameters']['embedding']['nout'] = 128
    #config_dict['model_parameters']['embedding']['num_blocks'] = 1
    #config_dict['model_parameters']['embedding']['datalength'] = 500
    #config_dict['model_parameters']['embedding']['middle_features'] = 1024
    # use_psd = True, middle_features = 512
    #config_dict['model_parameters']['embedding']['model'] = 'EmbeddingConv1D'
    #config_dict['model_parameters']['embedding']['ndet'] = 3
    #config_dict['model_parameters']['embedding']['nout'] = 128
    #config_dict['model_parameters']['embedding']['num_blocks'] = 3
    #config_dict['model_parameters']['embedding']['middle_channel'] = 32
    #config_dict['model_parameters']['embedding']['use_psd'] = False

    config_dict['model_parameters']['embedding']['model'] = 'EmbeddingResConv1DMLP'
    NCOND = 128
    config_dict['model_parameters']['embedding']['nout'] = NCOND
    #config_dict['model_parameters']['embedding']['ndet'] = 3
    config_dict['model_parameters']['embedding']['nbasis'] = config_dict['model_parameters']['Nbasis']
    config_dict['model_parameters']['embedding']['conv_params'] = {
            'in_channel':  [6,  128, 128, 128, 64, 64, 32, 32, 16, 16, 8, 8],
            'out_channel':     [128, 128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4],
            'kernel_size': [16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 1],
            'stride':      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'padding':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'dilation':    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'dropout':     [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    config_dict['model_parameters']['embedding']['mlp_params'] = {
            'in_features': [0, 512, 256],
            'out_features': [512, 256, NCOND],
        }


    # resnet after embedding, before flow
    #config_dict['model_parameters']['embedding_resnet'] = {}
    #config_dict['model_parameters']['embedding_resnet']['model'] = 'ResnetMLP1D'
    #config_dict['model_parameters']['embedding_resnet']['nout'] = 128
    #config_dict['model_parameters']['embedding_resnet']['num_blocks'] = 2
    #config_dict['model_parameters']['embedding_resnet']['in_feature'] = 128
    #config_dict['model_parameters']['embedding_resnet']['middle_features'] = 256
    # nout, num_blocks, in_feature, middle_features = 128

    # flow
    config_dict['model_parameters']['flow'] = {}

    config_dict['model_parameters']['flow']['model'] = 'CouplingNSF'
    config_dict['model_parameters']['flow']['n_inputs'] = 17
    config_dict['model_parameters']['flow']['n_transforms'] = 15
    config_dict['model_parameters']['flow']['n_conditional_inputs'] = NCOND
    config_dict['model_parameters']['flow']['n_neurons'] = 128  # 32 by default
    config_dict['model_parameters']['flow']['batch_norm_between_transforms'] = True
    config_dict['model_parameters']['flow']['batch_norm_within_blocks'] = True
    config_dict['model_parameters']['flow']['n_blocks_per_transform'] = 5  # 2 by default, 5
    config_dict['model_parameters']['flow']['num_bins'] = 8  # 4 by default, 8
    config_dict['model_parameters']['flow']['tail_bound'] = 1 # 5 by default, 1




    with open(f"{ckpt_dir}/config.json", 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
