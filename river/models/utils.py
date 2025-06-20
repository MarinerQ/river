import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
import bilby
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from glasflow import CouplingNSF, RealNVP
import zuko
from collections import OrderedDict, namedtuple
from itertools import product
import scipy

import pandas as pd
from .embedding.conv import EmbeddingConv1D, EmbeddingConv2D, MyEmbeddingConv2D,MyEmbeddingConv1D, EmbeddingResConv1DMLP, EmbeddingConv1DMLP
from .embedding.mlp import MLPResNet #EmbeddingMLP1D, ResnetMLP1D
from .embedding.simple_vit_1d import SimpleViT

LEGEND_LABEL_MAP = {'chirp_mass': r"$\mathcal{M}$",
                   'mass_ratio': r"$q$",
                   'a_1': r"$a_1$",
                   'a_2': r"$a_2$",
                   'tilt_1': r"$\theta_1$",
                   'tilt_2': r"$\theta_2$",
                   'phi_12': r"$\varphi_{12}$",
                   'phi_jl': r"$\varphi_{JL}$",
                   'lambda_tilde': r"$\tilde{\Lambda}$",
                   'delta_lambda_tilde': r"$\delta \tilde{\Lambda}$",
                   'theta_jn': r"$\theta_{JN}$",
                   'luminosity_distance': r"$d_L$",
                   'ra': r"$\alpha$",
                   'dec': r"$\delta$",
                   'psi': r"$\psi$",
                   'phase': r"$\phi_c$",
                   'geocent_time': r"$t_c$",
                   'mass_1': r"$m_1$",
                   'mass_2': r"$m_2$",
                   'lambda_1': r"$\Lambda_1$",
                   'lambda_2': r"$\Lambda_2$",}

############################################
########## Data Loading functions ##########
############################################
def save_model(filename, model):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {filename}')

def load_model(filename, verbose=False):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    if verbose:
        print(f'Model loaded from {filename}')
    return model 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adjust_lr(optimizer, gamma):
    for g in optimizer.param_groups:
        g['lr'] = g['lr']*gamma

def save_loss_data(train_losses, valid_losses, outdir, logscale='true', test_losses=None):
    np.savetxt(f'{outdir}/train_losses.txt', train_losses)
    np.savetxt(f'{outdir}/valid_losses.txt', valid_losses)

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    if test_losses is not None:
        plt.plot(test_losses, label='test')
        np.savetxt(f'{outdir}/test_losses.txt', test_losses)
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')

    ymin = min(min(train_losses), min(valid_losses)) - 3
    plt.ylim(ymin, max(train_losses))

    #if logscale and ymin>0:
    #    plt.yscale('log')
    #if len(train_losses)>100:
    #    plt.xscale('log')
    
    #plt.ylim(3, 1.2*max(train_losses))
    plt.savefig(f'{outdir}/losses.png')
    plt.close ('all')
    

def get_embd_dim(embd):
    if type(embd) in [EmbeddingConv1D, EmbeddingMLP1D]:
        dim = 1
    elif type(embd) in [EmbeddingConv2D]:
        dim = 2
    else:
        raise Exception("Wrong embedding layer type.")
    return dim 

def get_model(config_dict):
    config_dict_cpy = config_dict.copy()
    model_name = config_dict_cpy.pop('model')
    model_classes = {
        'SimpleViT': SimpleViT,
        'EmbeddingConv1D': EmbeddingConv1D,
        'EmbeddingResConv1DMLP': EmbeddingResConv1DMLP,
        'EmbeddingConv1DMLP': EmbeddingConv1DMLP,
        'MyEmbeddingConv2D': MyEmbeddingConv2D,
        'MyEmbeddingConv1D': MyEmbeddingConv1D,
        'EmbeddingConv2D': EmbeddingConv2D,
        #'EmbeddingMLP1D': EmbeddingMLP1D,
        #'ResnetMLP1D': ResnetMLP1D,
        'MLPResNet': MLPResNet,
        'CouplingNSF': CouplingNSF,
        'NSF': zuko.flows.NSF,
        'CNF': zuko.flows.CNF
    }
    if model_name in model_classes:
        return model_classes[model_name](**config_dict_cpy)
    else:
        raise Exception("Model not implemented!")

def get_train_func(flow):
    if type(flow) in [CouplingNSF, RealNVP]:
        return train_glasflow, eval_glasflow
    elif type(flow) in [zuko.flows.NSF, zuko.flows.CNF]:
        return train_zukoflow, eval_zukoflow
    else:
        raise Exception("Model not implemented!")

def project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen, project=True, downsample_rate=1, dim=1):
    '''
    strain: DatasetStrainFD in batches, e.g. DatasetStrainFD[0:10]
    psd: strain-like
    detector_names: DatasetStrainFD.detector_names
    ipca_gen: IPCAGenerator
    '''
    strain_amp = np.abs(strain)
    strain_phi = np.unwrap(np.angle(strain) , axis=-1)
    strain_real = np.real(strain)
    strain_imag = np.imag(strain)

    n_components = ipca_gen.n_components
    batch_size = strain.shape[0]
    ndet = len(detector_names)

    output_amp = []
    output_phi = []
    output_psd = []
    for i,detname in enumerate(detector_names):
        if project:
            output_amp.append(ipca_gen.project(strain_amp[:,i,:], detname, 'amplitude'))
            output_phi.append(ipca_gen.project(strain_phi[:,i,:], detname, 'phase'))
            output_psd.append(ipca_gen.project(psd[:,i,:], detname, 'amplitude'))
        else:
            output_amp.append(strain_amp.numpy()[:,i,:][:,::downsample_rate])
            output_phi.append(strain_phi[:,i,:][:,::downsample_rate])
            #output_amp.append(strain_real.numpy()[:,i,:][:,::downsample_rate])
            #output_phi.append(strain_imag.numpy()[:,i,:][:,::downsample_rate])
            output_psd.append(psd.numpy()[:,i,:][:,::downsample_rate])

    output_amp = torch.from_numpy(np.array(output_amp))
    output_phi = torch.from_numpy(np.array(output_phi))
    output_psd = torch.from_numpy(np.array(output_psd))
    data_length = output_amp.shape[-1]
    if dim==1:
        return torch.cat((output_amp, output_phi, output_psd)).movedim(0,1).float()
    elif dim==2:
        return torch.cat((output_amp, output_phi, output_psd)).movedim(0,1).float().view((batch_size,3,ndet,data_length))


def get_condition_2proj(embedding_proj, embedding_noproj, strain, psd, detector_names, ipca_gen, device, downsample_rate):
    inputs_proj = project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen, dim=get_embd_dim(embedding_proj)).to(device)
    inputs_noproj = project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen, project=False, downsample_rate=downsample_rate, dim=get_embd_dim(embedding_noproj)).to(device) 

    embedding_out_proj = embedding_proj(inputs_proj)
    embedding_out_noproj = embedding_noproj(inputs_noproj)
    condition = torch.cat((embedding_out_proj, embedding_out_noproj), -1)#.to(device)

    return condition

def get_condition_1proj(embedding, strain, psd, detector_names, ipca_gen, device, downsample_rate):
    #inputs= project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen, dim=get_embd_dim(embedding)).to(device)
    inputs = project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen, project=False, downsample_rate=downsample_rate, dim=get_embd_dim(embedding)).to(device) 

    embedding_out = embedding(inputs)
    #embedding_out_noproj = embedding_noproj(inputs_noproj)
    #condition = torch.cat((embedding_out_proj, embedding_out_noproj), -1)#.to(device)

    return embedding_out.to(device)

def get_condition_1projres(embedding, resnet, strain, psd, detector_names, ipca_gen, device, downsample_rate):
    #inputs= project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen, dim=get_embd_dim(embedding)).to(device)
    inputs = project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen, project=False, downsample_rate=downsample_rate, dim=get_embd_dim(embedding)).to(device) 

    embedding_out = embedding(inputs)
    resout = resnet(embedding_out)
    #embedding_out_noproj = embedding_noproj(inputs_noproj)
    #condition = torch.cat((embedding_out_proj, embedding_out_noproj), -1)#.to(device)

    return resout.to(device)

##################################################
########## Train, valid, test functions ##########
##################################################
def train_zukoflow(flow, embedding_proj, embedding_noproj, optimizer, dataloader, detector_names, ipca_gen, device='cpu',downsample_rate=1):
    flow.train()
    embedding_proj.train()
    embedding_noproj.train()
    loss_list = []
    for theta, strain, psd in dataloader:
        optimizer.zero_grad()
        '''
        inputs_proj = project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen).to(device)
        inputs_noproj = project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen, project=False).unsqueeze(1)[:,:,::downsample_rate].to(device)
        theta = theta.to(device)

        embedding_out_proj = embedding_proj(inputs_proj)
        embedding_out_noproj = embedding_noproj(inputs_noproj)
        condition = torch.cat((embedding_out_proj, embedding_out_noproj), -1)
        '''
        theta = theta.to(device)
        condition = get_condition_2proj(embedding_proj, embedding_noproj, strain, psd, detector_names, ipca_gen, device, downsample_rate)

        loss = -flow(condition).log_prob(theta).mean() # mean(list of losses of elements in this batch)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.detach())

    mean_loss = torch.stack(loss_list).mean().item() # mean(list of mean losses of each batch)
    std_loss = torch.stack(loss_list).std().item()
    return mean_loss, std_loss

def eval_zukoflow(flow, embedding_proj, embedding_noproj, dataloader, detector_names, ipca_gen, device='cpu',downsample_rate=1):
    flow.eval()
    embedding_proj.eval()
    embedding_noproj.eval()
    loss_list = []
    with torch.no_grad():
        for theta, strain, psd in dataloader:
            theta = theta.to(device)
            condition = get_condition_2proj(embedding_proj, embedding_noproj, strain, psd, detector_names, ipca_gen, device, downsample_rate)

            loss = -flow(condition).log_prob(theta).mean()
            loss_list.append(loss.detach())

    mean_loss = torch.stack(loss_list).mean().item()
    std_loss = torch.stack(loss_list).std().item()
    return mean_loss, std_loss


def sample_zukoflow(flow, embedding_proj, embedding_noproj, dataset, detector_names, ipca_gen, device='cpu', Nsample=5000, downsample_rate=1):
    flow.eval()
    embedding_proj.eval()
    embedding_noproj.eval()
    #loss_list = []
    #sample_list = []
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for theta, strain, psd in dataloader:
            theta = theta.to(device)
            condition = get_condition_2proj(embedding_proj, embedding_noproj, strain, psd, detector_names, ipca_gen, device, downsample_rate)

            loss = -flow(condition).log_prob(theta)
            samples = flow(condition).sample((Nsample,))

            #loss_list.append(loss.detach())
            #sample_list.append(samples)

    return samples.movedim(0,1).movedim(1,2), loss

'''
def train_GlasNSFWarpper(model, optimizer, dataloader, device='cpu', minibatch_size=1, logger=None,extra_condition=None):
    model.train()
    loss_list = []
    nbatch=0
    for theta, x in dataloader:
        optimizer.zero_grad()
        theta = theta.to(device)
        x = x.to(device)
        
        if minibatch_size>1:
            # x: [bs, minibatch_size, nchannel, nbasis]
            # theta: [bs, minibatch_size, npara]
            bs = x.shape[0]
            nbasis = x.shape[-1]
            nchannel = x.shape[-2]
            npara = theta.shape[-1]
            theta = theta.view(bs*minibatch_size, npara)
            x = x.view(bs*minibatch_size, nchannel, nbasis)
        if extra_condition:
            extra_condition = extra_condition.to(device)
            loss = -model.log_prob(theta, x, extra_condition).mean()
        else:
            loss = -model.log_prob(theta, x).mean()
        loss.backward()
        optimizer.step()
        if logger and nbatch%100==0:
            logger.info(f'----batch {nbatch}, train loss = {loss.detach()}.----')
        loss_list.append(loss.detach())
        nbatch+=1

    mean_loss = torch.stack(loss_list).mean().item() # mean(list of mean losses of each batch)
    std_loss = torch.stack(loss_list).std().item()
    return mean_loss, std_loss

def eval_GlasNSFWarpper(model, dataloader, device='cpu', minibatch_size=1, logger=None, extra_condition=None):
    model.eval()
    loss_list = []
    nbatch = 0
    with torch.no_grad():
        for theta, x in dataloader:
            theta = theta.to(device)
            x = x.to(device)
            if minibatch_size>1:
                # x: [bs, minibatch_size, nchannel, nbasis]
                # theta: [bs, minibatch_size, npara]
                bs = x.shape[0]
                nbasis = x.shape[-1]
                nchannel = x.shape[-2]
                npara = theta.shape[-1]
                theta = theta.view(bs*minibatch_size, npara)
                x = x.view(bs*minibatch_size, nchannel, nbasis)
            if extra_condition:
                extra_condition = extra_condition.to(device)
                loss = -model.log_prob(theta, x, extra_condition).mean()
            else:
                loss = -model.log_prob(theta, x).mean()

            if logger and nbatch%100==0:
                logger.info(f'----batch {nbatch}, valid loss = {loss.detach()}.----')
            loss_list.append(loss.detach())
            nbatch += 1

    mean_loss = torch.stack(loss_list).mean().item()
    std_loss = torch.stack(loss_list).std().item()
    return mean_loss, std_loss


def sample_GlasNSFWarpper(model, dataset, device='cpu', Nsample=5000, extra_condition=None):
    model.eval()
    #loss_list = []
    #sample_list = []
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for theta, x in dataloader:
            theta = theta.to(device)
            x = x.to(device)
            #x = project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen).to(device)
            if extra_condition:
                extra_condition = extra_condition.to(device)
                loss = -model.log_prob(theta, x, extra_condition).mean()
                samples = model.sample((Nsample,), x, extra_condition)
            else:
                loss = -model.log_prob(theta, x).mean()
                samples = model.sample((Nsample,), x)

            #loss_list.append(loss.detach())
            #sample_list.append(samples)

    return samples.movedim(0,1).movedim(1,2), loss
'''

def make_results(sample_list, injection_parameters_list, parameter_names):
    Nresult = len(sample_list)

    result_list = []
    for i in range(Nresult):
        result = bilby.gw.result.CompactBinaryCoalescenceResult()
        injection_parameters = {}
        posterior_dict = {}
        search_parameter_keys = []
        parameter_labels_with_unit = []
        for j,paraname in enumerate(parameter_names):
            injection_parameters[paraname] = injection_parameters_list[paraname][i]
            posterior_dict[paraname] = np.array(sample_list[i][j].cpu())
            search_parameter_keys.append(paraname)
            parameter_labels_with_unit.append(paraname)
        
        result.posterior = pd.DataFrame.from_dict(posterior_dict)
        result.injection_parameters = injection_parameters
        result.search_parameter_keys = search_parameter_keys
        result.parameter_labels_with_unit = parameter_labels_with_unit
        result_list.append(result)
    
    return result_list

def make_prior(injection_parameters_list, parameter_names):
    result = bilby.gw.result.CompactBinaryCoalescenceResult()
    injection_parameters = {}
    posterior_dict = {}
    search_parameter_keys = []
    parameter_labels_with_unit = []
    for j,paraname in enumerate(parameter_names):
        #injection_parameters[paraname] = injection_parameters_list[paraname][i]
        posterior_dict[paraname] = injection_parameters_list[paraname]
        search_parameter_keys.append(paraname)
        parameter_labels_with_unit.append(paraname)
    
    result.posterior = pd.DataFrame.from_dict(posterior_dict)
    #result.injection_parameters = injection_parameters
    result.search_parameter_keys = search_parameter_keys
    result.parameter_labels_with_unit = parameter_labels_with_unit

    return result

def make_pp_plot(results, filename=None, save=True, confidence_interval=[0.68, 0.95, 0.997],
                 lines=None, fontsize=20, keys=None, title=True,
                 confidence_interval_alpha=0.1, weight_list=None, legend_label_map=LEGEND_LABEL_MAP,
                 **kwargs):
    """
    Make a P-P plot for a set of runs with injected signals.
    Copied from bilby but fixed some bugs.
    """
    import matplotlib.pyplot as plt

    if keys is None:
        keys = results[0].search_parameter_keys

    if weight_list is None:
        weight_list = [None] * len(results)

    credible_levels = pd.DataFrame()
    for i, result in enumerate(results):
        #credible_levels = credible_levels.append(
        #    result.get_all_injection_credible_levels(keys, weights=weight_list[i]),
        #    ignore_index=True)
        #print(type( result.get_all_injection_credible_levels(keys, weights=weight_list[i])))
        credible_levels = pd.concat([credible_levels, pd.DataFrame.from_records([result.get_all_injection_credible_levels(keys, weights=weight_list[i])])], ignore_index=True)

    if lines is None:
        colors = ["C{}".format(i) for i in range(8)]
        linestyles = ["-", "--", ":"]
        lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    if len(lines) < len(credible_levels.keys()):
        raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)
    fig, ax = plt.subplots(figsize=(12,12))

    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(confidence_interval)
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval")

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')

    pvalues = []
    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        pvalue = scipy.stats.kstest(credible_levels[key], 'uniform').pvalue
        pvalues.append(pvalue)


        name = key
        label = r"{} ({:2.3f})".format(name, pvalue)
        if legend_label_map and name in legend_label_map.keys():
            label = label.replace(name, legend_label_map[name])
        plt.plot(x_values, pp, lines[ii], label=label, **kwargs)

    Pvals = namedtuple('pvals', ['combined_pvalue', 'pvalues', 'names'])
    pvals = Pvals(combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
                  pvalues=pvalues,
                  names=list(credible_levels.keys()))

    if title:
        ax.set_title("N={}, p-value={:2.4f}".format(
            len(results), pvals.combined_pvalue), fontsize=fontsize)
    ax.set_xlabel("C.I.", fontsize=fontsize)
    ax.set_ylabel("Fraction of events in C.I.", fontsize=fontsize)
    ax.legend(handlelength=2, labelspacing=0.25, fontsize=fontsize, ncol=2, loc=(0.01,0.68))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.tight_layout()
    if save:
        fig.savefig(filename)

    return fig, pvals