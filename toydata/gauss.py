import numpy as np
import torch

def normal_gauss_ds(
        n_features:int = 2, 
        n_samples:int = 100,
        means:list[float] = [-0.5, 0.5],
        stds:list[float] = [1, 1],
        dtype: torch.dtype = torch.float32, 
        device: str = 'cpu'):
    
    if len(means) != len(stds):
        raise ValueError(f'len of means {len(means)} and stds {len(stds)} need to be equal')
    
    if device not in ['cpu', 'cuda']:
        raise ValueError(f"device need to be in ['cpu', 'device']")

    if n_features <= 0:
        raise ValueError(f'n_features {n_features} need to be larger than 0')

    clusters = []

    for idx, (mean, std) in enumerate(zip(means, stds)):

        feature = torch.normal(mean=mean, std=std, size=(n_samples, n_features))
        category = torch.full(fill_value=idx, size=(n_samples, 1))

        feature.to(device=device, dtype=dtype)
        category.to(device=device, dtype=torch.int)

        data = torch.cat([feature, category], dim=1)

        clusters.append(data)
    
    stacked_data = torch.cat(clusters)

    return stacked_data