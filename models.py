from ols_denoise_model import OLSDenoiseSystem
from mean_med_denoise_model import MeanMedDenoiseSystem
from dncnn_denoise_model import DnCNNDenoiseSystem

def get_model(model):
    if model == 'ols':
        return OLSDenoiseSystem
    elif model == 'meanmed':
        return MeanMedDenoiseSystem
    elif model == 'dncnn':
        return DnCNNDenoiseSystem
    else:
        raise ValueError('Unknown model')
