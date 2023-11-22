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
    config_dict['data_generator_parameters']['f_low'] = 20
    config_dict['data_generator_parameters']['f_ref'] = 20
    config_dict['data_generator_parameters']['sampling_frequency'] = 2048
    config_dict['data_generator_parameters']['waveform_approximant'] = 'IMRPhenomPv2_NRTidal'
    config_dict['data_generator_parameters']['parameter_names'] = PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
    config_dict['data_generator_parameters']['context_parameter_names'] = PARAMETER_NAMES_CONTEXT_PRECESSINGBNS_BILBY
    config_dict['data_generator_parameters']['PSD_type'] = 'bilby_default'
    config_dict['data_generator_parameters']['use_sealgw_detector'] = True
    config_dict['data_generator_parameters']['snr_threshold'] = 8
    config_dict['data_generator_parameters']['selection_factor'] = 2
    config_dict['data_generator_parameters']['a_max'] = 0.1
    config_dict['data_generator_parameters']['d_min'] = 10
    config_dict['data_generator_parameters']['d_max'] = 200
    config_dict['data_generator_parameters']['d_power'] = 3
    config_dict['data_generator_parameters']['tc_min'] = -0.1
    config_dict['data_generator_parameters']['tc_max'] = 0.1

    # training_parameters
    config_dict['training_parameters'] = {}
    config_dict['training_parameters']['Nsample'] = 2000
    config_dict['training_parameters']['Nvalid'] = 1000
    config_dict['training_parameters']['batch_size_train'] = 512
    config_dict['training_parameters']['batch_size_valid'] = 256
    config_dict['training_parameters']['device'] = 'cuda:0'
    config_dict['training_parameters']['lr'] = 5e-4
    config_dict['training_parameters']['gamma'] = 0.7
    config_dict['training_parameters']['max_epoch'] = 10000
    config_dict['training_parameters']['epoches_update'] = 5
    config_dict['training_parameters']['epoches_pretrain'] = 10
    config_dict['training_parameters']['epoches_save_loss'] = 5
    config_dict['training_parameters']['epoches_adjust_lr'] = 10
    config_dict['training_parameters']['epoches_adjust_lr_again'] = 8
    config_dict['training_parameters']['load_from_previous_train'] = False



    # model_parameters
    config_dict['model_parameters'] = {}
    # IPCA
    # IPCA_BNSFD_10000to500_ExpUnwrap_fixtc_highspin_200Mpc
    # IPCA_BNSFD_10000to500_ExpUnwrap_fixtc
    #config_dict['model_parameters']['ipca_path'] = '../scripts/ipca_models/IPCA_BNSFD_10000to500_ExpUnwrap_fixtc_highspin_200Mpc.pickle'
    config_dict['model_parameters']['ipca_path'] = '../scripts/ipca_models/IPCA_BNSFD_10000to500_ExpUnwrap_fixtc_lowspin_200Mpc.pickle'

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
    #config_dict['model_parameters']['embedding_noproj']['model'] = 'EmbeddingConv2D'
    #config_dict['model_parameters']['embedding_noproj']['nout'] = 128
    #config_dict['model_parameters']['embedding_noproj']['num_blocks'] = 3
    #config_dict['model_parameters']['embedding_noproj']['downsample_rate'] = 4
    #config_dict['model_parameters']['embedding_noproj']['middle_channel'] = 18
    config_dict['model_parameters']['embedding_noproj']['model'] = 'EmbeddingConv1D'
    config_dict['model_parameters']['embedding_noproj']['ndet'] = 3
    config_dict['model_parameters']['embedding_noproj']['downsample_rate'] = 2 # if no proj
    config_dict['model_parameters']['embedding_noproj']['nout'] = 128
    config_dict['model_parameters']['embedding_noproj']['num_blocks'] = 5
    config_dict['model_parameters']['embedding_noproj']['middle_channel'] = 18
    # use_psd = True, middle_channel = 512, kernel_size=1, stride=1, padding=0, dilation=1

    # flow
    config_dict['model_parameters']['flow'] = {}
    config_dict['model_parameters']['flow']['model'] = 'CouplingNSF'
    config_dict['model_parameters']['flow']['n_inputs'] = 17 
    config_dict['model_parameters']['flow']['n_transforms'] = 128
    config_dict['model_parameters']['flow']['n_conditional_inputs'] = 256
    config_dict['model_parameters']['flow']['n_neurons'] = 256
    config_dict['model_parameters']['flow']['batch_norm_between_transforms'] = True


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