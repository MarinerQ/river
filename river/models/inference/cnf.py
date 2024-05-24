import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import zuko
from glasflow import RealNVP, CouplingNSF

from ..utils import get_model, project_strain_data_FDAPhi

class GlasNSFConv1DRes(nn.Module):
    def __init__(self, config_dict, **kwargs):
        super().__init__()
        self.config_dict = config_dict
        self.config_flow = config_dict['model_parameters']['flow']
        self.config_embedding =  config_dict['model_parameters']['embedding_proj']
        self.config_resnet =  config_dict['model_parameters']['embedding_resnet']

        self.embedding = get_model(self.config_embedding)
        self.resnet = get_model(self.config_resnet)
        self.flow = get_model(self.config_flow)

    def condition(self, x):
        embedding_out = self.embedding(x)
        resout = self.resnet(embedding_out)
        return resout
    
    def forward(self, theta,  x):
        condition = self.condition(x)
        return self.flow._transform.forward(theta, context=condition)

    def sample(self, num_samples, x):
        condition = self.condition(x)

        #lencond = condition.shape[-1]
        #samples = self.flow._transform.inverse(num_samples, conditional=condition.expand((num_samples,lencond )))
        lencond = condition.shape[-1]
        
        noise = self.flow._distribution.sample(num_samples)
        samples, _ = self.flow._transform.inverse(noise, context=condition.expand((num_samples,lencond )))
        return samples

    def log_prob(self, theta, x):
        condition = self.condition(x)
        noise, logabsdet = self.flow._transform(theta, context=condition)
        log_prob = self.flow._distribution.log_prob(noise)
        return log_prob + logabsdet
        

class GlasNSFConv1D(nn.Module):
    def __init__(self, config_dict, **kwargs):
        super().__init__()
        self.config_dict = config_dict
        self.config_flow = config_dict['model_parameters']['flow']
        self.config_embedding =  config_dict['model_parameters']['embedding']

        self.embedding = get_model(self.config_embedding)
        self.flow = get_model(self.config_flow)

    def condition(self, x):
        embedding_out = self.embedding(x)
        #embedding_out = x[:, 0, :]
        return embedding_out
    
    def forward(self, theta,  x):
        condition = self.condition(x)
        return self.flow._transform.forward(theta, context=condition)

    def sample(self, num_samples, x):
        condition = self.condition(x)

        #lencond = condition.shape[-1]
        #samples = self.flow._transform.inverse(num_samples, conditional=condition.expand((num_samples,lencond )))
        lencond = condition.shape[-1]
        
        noise = self.flow._distribution.sample(num_samples)
        samples, _ = self.flow._transform.inverse(noise, context=condition.expand((num_samples,lencond )))
        return samples

    def log_prob(self, theta, x):
        condition = self.condition(x)
        #theta_2d = theta[:,0:2]
        noise, logabsdet = self.flow._transform(theta, context=condition)
        log_prob = self.flow._distribution.log_prob(noise)
        return log_prob + logabsdet
        

class GlasflowEmbdding(nn.Module):
    def __init__(self, config_dict, **kwargs):
        super().__init__()
        self.config_dict = config_dict
        self.config_flow = config_dict['model_parameters']['flow']
        self.config_embedding =  config_dict['model_parameters']['embedding']

        self.embedding = get_model(self.config_embedding)
        self.flow = get_model(self.config_flow)

    def condition(self, x):
        embedding_out = self.embedding(x)
        return embedding_out
    
    def forward(self, theta,  x):
        condition = self.condition(x)
        return self.flow._transform.forward(theta, context=condition)

    def sample(self, num_samples, x):
        condition = self.condition(x)

        #lencond = condition.shape[-1]
        #samples = self.flow._transform.inverse(num_samples, conditional=condition.expand((num_samples,lencond )))
        lencond = condition.shape[-1]
        
        noise = self.flow._distribution.sample(num_samples)
        samples, _ = self.flow._transform.inverse(noise, context=condition.expand((num_samples,lencond )))
        return samples

    def log_prob(self, theta, x):
        condition = self.condition(x)
        noise, logabsdet = self.flow._transform(theta, context=condition)
        log_prob = self.flow._distribution.log_prob(noise)
        return log_prob + logabsdet


class GlasflowEmbeddingExtraCondition(nn.Module):
    def __init__(self, config_dict, **kwargs):
        super().__init__()
        self.config_dict = config_dict
        self.config_flow = config_dict['model_parameters']['flow']
        self.config_embedding =  config_dict['model_parameters']['embedding']

        self.embedding = get_model(self.config_embedding)
        self.flow = get_model(self.config_flow)

    def condition(self, x, extra_condition):
        embedding_out = self.embedding(x)
        combined_condition = torch.cat((embedding_out, extra_condition), dim=-1)
        return combined_condition
    
    def forward(self, theta,  x, extra_condition):
        condition = self.condition(x, extra_condition)
        return self.flow._transform.forward(theta, context=condition)

    def sample(self, num_samples, x, extra_condition):
        condition = self.condition(x, extra_condition)

        #lencond = condition.shape[-1]
        #samples = self.flow._transform.inverse(num_samples, conditional=condition.expand((num_samples,lencond )))
        lencond = condition.shape[-1]
        
        noise = self.flow._distribution.sample(num_samples)
        samples, _ = self.flow._transform.inverse(noise, context=condition.expand((num_samples,lencond )))
        return samples

    def log_prob(self, theta, x, extra_condition):
        condition = self.condition(x, extra_condition)
        noise, logabsdet = self.flow._transform(theta, context=condition)
        log_prob = self.flow._distribution.log_prob(noise)
        return log_prob + logabsdet