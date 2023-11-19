import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
import bilby
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pandas as pd
from .embedding.pca import project_strain_data_FDAPhi


def save_model(filename, model):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {filename}')

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f'Model loaded from {filename}')
    return model 

def adjust_lr(optimizer, gamma):
    for g in optimizer.param_groups:
        g['lr'] = g['lr']*gamma

def save_loss_data(train_losses, valid_losses, outdir, logscale='true'):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')

    if logscale:
        plt.yscale('log')
    
    plt.savefig(f'{outdir}/losses.png')
    np.savetxt(f'{outdir}/train_losses.txt', train_losses)
    np.savetxt(f'{outdir}/valid_losses.txt', valid_losses)


def get_condition_2proj(embedding_proj, embedding_noproj, theta, strain, psd, detector_names, ipca_gen, device, downsample_rate):
    inputs_proj = project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen).to(device)
    inputs_noproj = project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen, project=False, downsample_rate=downsample_rate).to(device) #.unsqueeze(1)
    theta = theta.to(device)

    embedding_out_proj = embedding_proj(inputs_proj)
    embedding_out_noproj = embedding_noproj(inputs_noproj)
    condition = torch.cat((embedding_out_proj, embedding_out_noproj), -1)

    return condition

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
        condition = get_condition_2proj(embedding_proj, embedding_noproj, theta, strain, psd, detector_names, ipca_gen, device, downsample_rate)

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
            condition = get_condition_2proj(embedding_proj, embedding_noproj, theta, strain, psd, detector_names, ipca_gen, device, downsample_rate)

            loss = -flow(condition).log_prob(theta).mean()
            loss_list.append(loss.detach())

    mean_loss = torch.stack(loss_list).mean().item()
    std_loss = torch.stack(loss_list).std().item()
    return mean_loss, std_loss


def sample_zukoflow(flow, embedding_proj, embedding_noproj, dataset, detector_names, ipca_gen, device='cpu', Nsample=5000, downsample_rate=1):
    flow.eval()
    #loss_list = []
    #sample_list = []
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for theta, strain, psd in dataloader:
            condition = get_condition_2proj(embedding_proj, embedding_noproj, theta, strain, psd, detector_names, ipca_gen, device, downsample_rate)

            loss = -flow(condition).log_prob(theta)
            samples = flow(condition).sample((Nsample,))

            #loss_list.append(loss.detach())
            #sample_list.append(samples)

    return samples.movedim(0,1).movedim(1,2), loss


def train_glasflow(flow, embedding_proj, embedding_noproj, optimizer, dataloader, detector_names, ipca_gen, device='cpu',downsample_rate=1):
    flow.train()
    embedding_proj.train()
    embedding_noproj.train()
    loss_list = []
    for theta, strain, psd in dataloader:
        optimizer.zero_grad()

        condition = get_condition_2proj(embedding_proj, embedding_noproj, theta, strain, psd, detector_names, ipca_gen, device, downsample_rate)

        loss = -flow.log_prob(theta, conditional=condition).mean() # mean(list of losses of elements in this batch)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.detach())

    mean_loss = torch.stack(loss_list).mean().item() # mean(list of mean losses of each batch)
    std_loss = torch.stack(loss_list).std().item()
    return mean_loss, std_loss

def eval_glasflow(flow, embedding_proj, embedding_noproj, dataloader, detector_names, ipca_gen, device='cpu',downsample_rate=1):
    flow.eval()
    embedding_proj.eval()
    embedding_noproj.eval()
    loss_list = []
    with torch.no_grad():
        for theta, strain, psd in dataloader:

            condition = get_condition_2proj(embedding_proj, embedding_noproj, theta, strain, psd, detector_names, ipca_gen, device, downsample_rate)

            loss = -flow.log_prob(theta, conditional=condition).mean()
            loss_list.append(loss.detach())

    mean_loss = torch.stack(loss_list).mean().item()
    std_loss = torch.stack(loss_list).std().item()
    return mean_loss, std_loss


def sample_glasflow(flow, embedding_proj, embedding_noproj, dataset, detector_names, ipca_gen, device='cpu', Nsample=5000,downsample_rate=1):
    flow.eval()
    #loss_list = []
    #sample_list = []
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for theta, strain, psd in dataloader:
            condition = get_condition_2proj(embedding_proj, embedding_noproj, theta, strain, psd, detector_names, ipca_gen, device, downsample_rate)

            loss = -flow.log_prob(theta, conditional=condition).mean()
            samples = flow.sample(Nsample, conditional=condition)

            #loss_list.append(loss.detach())
            #sample_list.append(samples)

    return samples.movedim(0,1).movedim(1,2), loss


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