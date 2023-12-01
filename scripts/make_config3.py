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
    config_dict['data_generator_parameters']['duration'] = 320
    config_dict['data_generator_parameters']['f_low'] = 20
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
    config_dict['data_generator_parameters']['ipca'] = '/home/qian.hu/mlpe/river/scripts/ipca_models/hphc/IPCA_HPHC_BNSFD_10000to256_ExpUnwrap_lowspin_200Mpc.pickle'

    # pre-calculated waveform parameters
    config_dict['precalwf_parameters'] = {}
    config_dict['precalwf_parameters']['train'] = {}
    config_dict['precalwf_parameters']['train']['folder'] = '/home/qian.hu/mlpe/training_data/bns_320s_lowspin/train'
    config_dict['precalwf_parameters']['train']['nbatch'] = 1
    config_dict['precalwf_parameters']['train']['file_per_batch'] = 1
    config_dict['precalwf_parameters']['train']['sample_per_file'] = 100000 
    config_dict['precalwf_parameters']['train']['filename_prefix'] = 'precalwf_bns_320s_lowspin'

    config_dict['precalwf_parameters']['valid'] = {}
    config_dict['precalwf_parameters']['valid']['folder'] = '/home/qian.hu/mlpe/training_data/bns_320s_lowspin/valid'
    config_dict['precalwf_parameters']['valid']['nbatch'] = 1
    config_dict['precalwf_parameters']['valid']['file_per_batch'] = 1
    config_dict['precalwf_parameters']['valid']['sample_per_file'] = 1000
    config_dict['precalwf_parameters']['valid']['filename_prefix'] = 'precalwf_bns_320s_lowspin'



    # training_parameters
    config_dict['training_parameters'] = {}
    #config_dict['training_parameters']['Nsample'] = 10000
    #config_dict['training_parameters']['Nvalid'] = 500
    config_dict['training_parameters']['Ntest'] = 10
    config_dict['training_parameters']['batch_size_train'] = 4096
    config_dict['training_parameters']['batch_size_valid'] = 512
    config_dict['training_parameters']['batch_size_test'] = 10
    config_dict['training_parameters']['device'] = 'cuda:1'
    config_dict['training_parameters']['lr'] = 5e-4
    config_dict['training_parameters']['gamma'] = 0.7
    config_dict['training_parameters']['max_epoch'] = 10000
    config_dict['training_parameters']['epoches_pretrain'] = 10
    config_dict['training_parameters']['epoches_save_loss'] = 5
    config_dict['training_parameters']['epoches_adjust_lr'] = 15
    config_dict['training_parameters']['epoches_adjust_lr_again'] = 8
    config_dict['training_parameters']['load_from_previous_train'] = False



    # model_parameters
    config_dict['model_parameters'] = {}
    # IPCA
    # IPCA_BNSFD_10000to500_ExpUnwrap_fixtc_highspin_200Mpc
    # IPCA_BNSFD_10000to500_ExpUnwrap_fixtc
    config_dict['model_parameters']['ipca_path'] = '/home/qian.hu/mlpe/river/scripts/ipca_models/lhv/IPCA_BNSFD_10000to256_ExpUnwrap_fixtc_lowspin_200Mpc.pickle'

    # Embedding - projection
    config_dict['model_parameters']['embedding_proj'] = {}
    #config_dict['model_parameters']['embedding_proj']['model'] = 'EmbeddingMLP1D'
    #config_dict['model_parameters']['embedding_proj']['ndet'] = 3
    #config_dict['model_parameters']['embedding_proj']['nout'] = 128
    #config_dict['model_parameters']['embedding_proj']['num_blocks'] = 1
    #config_dict['model_parameters']['embedding_proj']['datalength'] = 500
    #config_dict['model_parameters']['embedding_proj']['middle_features'] = 1024
    # use_psd = True, middle_features = 512
    config_dict['model_parameters']['embedding_proj']['model'] = 'EmbeddingConv1D'
    config_dict['model_parameters']['embedding_proj']['ndet'] = 3
    config_dict['model_parameters']['embedding_proj']['nout'] = 128
    config_dict['model_parameters']['embedding_proj']['num_blocks'] = 3
    config_dict['model_parameters']['embedding_proj']['middle_channel'] = 32

    # Embedding - no projection. Remember to add downsample_rate!
    config_dict['model_parameters']['embedding_noproj'] = {}
    config_dict['model_parameters']['embedding_noproj']['model'] = 'EmbeddingConv2D'
    config_dict['model_parameters']['embedding_noproj']['nout'] = 128
    config_dict['model_parameters']['embedding_noproj']['num_blocks'] = 3
    config_dict['model_parameters']['embedding_noproj']['downsample_rate'] = 4
    config_dict['model_parameters']['embedding_noproj']['middle_channel'] = 18
    #config_dict['model_parameters']['embedding_noproj']['model'] = 'EmbeddingConv1D'
    #config_dict['model_parameters']['embedding_noproj']['ndet'] = 3
    #config_dict['model_parameters']['embedding_noproj']['downsample_rate'] = 2 # if no proj
    #config_dict['model_parameters']['embedding_noproj']['nout'] = 128
    #config_dict['model_parameters']['embedding_noproj']['num_blocks'] = 5
    #config_dict['model_parameters']['embedding_noproj']['middle_channel'] = 18
    # use_psd = True, middle_channel = 512, kernel_size=1, stride=1, padding=0, dilation=1


    # resnet after embedding, before flow
    config_dict['model_parameters']['embedding_resnet'] = {}
    config_dict['model_parameters']['embedding_resnet']['model'] = 'ResnetMLP1D'
    config_dict['model_parameters']['embedding_resnet']['nout'] = 128
    config_dict['model_parameters']['embedding_resnet']['num_blocks'] = 2
    config_dict['model_parameters']['embedding_resnet']['in_feature'] = 128
    config_dict['model_parameters']['embedding_resnet']['middle_features'] = 200
    # nout, num_blocks, in_feature, middle_features = 128

    # flow
    config_dict['model_parameters']['flow'] = {}

    config_dict['model_parameters']['flow']['model'] = 'CouplingNSF'
    config_dict['model_parameters']['flow']['n_inputs'] = 17 
    config_dict['model_parameters']['flow']['n_transforms'] = 128 # 
    config_dict['model_parameters']['flow']['n_conditional_inputs'] = 128
    config_dict['model_parameters']['flow']['n_neurons'] = 256
    config_dict['model_parameters']['flow']['batch_norm_between_transforms'] = True
    config_dict['model_parameters']['flow']['batch_norm_within_blocks'] = True
    #config_dict['model_parameters']['flow']['model'] = 'NSF'
    #config_dict['model_parameters']['flow']['features'] = 17
    #config_dict['model_parameters']['flow']['context'] = 256
    #config_dict['model_parameters']['flow']['transforms'] = 128
    #config_dict['model_parameters']['flow']['hidden_features'] = (640, 640)
    #zuko.flows.NSF(features=2, transforms=3, hidden_features=(64, 64)) context=1



    with open(f"{ckpt_dir}/config.json", 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)

'''
# EmbeddingConv1D example
config_dict['model_parameters']['embedding_noproj']['model'] = 'EmbeddingConv1D'
config_dict['model_parameters']['embedding_noproj']['ndet'] = 3
config_dict['model_parameters']['embedding_noproj']['downsample_rate'] = 4 # if no proj
config_dict['model_parameters']['embedding_noproj']['nout'] = 128
config_dict['model_parameters']['embedding_noproj']['num_blocks'] = 5
config_dict['model_parameters']['embedding_noproj']['middle_channel'] = 64
# use_psd = True, middle_channel = 512, kernel_size=1, stride=1, padding=0, dilation=1

# EmbeddingConv2D example
config_dict['model_parameters']['embedding_noproj']['model'] = 'EmbeddingConv2D'
config_dict['model_parameters']['embedding_noproj']['nout'] = 128
config_dict['model_parameters']['embedding_noproj']['num_blocks'] = 5
# use_psd = True, middle_channel = 16, kernel_size=1, stride=1, padding=0, dilation=1)

# EmbeddingMLP1D example
config_dict['model_parameters']['embedding_noproj']['model'] = 'EmbeddingMLP1D'
config_dict['model_parameters']['embedding_noproj']['ndet'] = 3
config_dict['model_parameters']['embedding_noproj']['nout'] = 128
config_dict['model_parameters']['embedding_noproj']['num_blocks'] = 5
config_dict['model_parameters']['embedding_noproj']['datalength'] = 500
# use_psd = True, middle_features = 512
'''