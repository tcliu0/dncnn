from ols_denoise_model import OLSDenoiseSystem
from mean_med_denoise_model import MeanMedDenoiseSystem
from dncnn_denoise_model import DnCNNDenoiseSystem
from identity_model import IdentitySystem
from median_model import MedianSystem

def get_model(model):
    if model == 'ols':
        return OLSDenoiseSystem
    elif model == 'meanmed':
        return MeanMedDenoiseSystem
    elif model == 'dncnn':
        return DnCNNDenoiseSystem
    elif model == 'identity':
        return IdentitySystem    
    elif model == 'median':
        return MedianSystem    
    else:
        raise ValueError('Unknown model')
