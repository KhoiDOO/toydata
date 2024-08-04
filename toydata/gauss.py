from toydata.utils import check_means_stds_1d, check_device, check_n_feature

import numpy as np
import torch


def normal_gauss_ds(
        n_features:int = 2, 
        n_samples:int = 100,
        means:list[float] = [-0.5, 0.5],
        stds:list[float] = [1, 1],
        device:str = 'cpu',
        shuffle:bool = True) -> torch.Tensor:
    
    check_means_stds_1d(means=means, stds=stds)
    check_device(device=device)
    check_n_feature(n_features=n_features)

    clusters = []

    for idx, (mean, std) in enumerate(zip(means, stds)):

        feature = torch.normal(mean=mean, std=std, size=(n_samples, n_features))
        category = torch.full(fill_value=idx, size=(n_samples, 1))

        feature.to(device=device, dtype=torch.float32)
        category.to(device=device, dtype=torch.int)

        data = torch.cat([feature, category], dim=1)

        clusters.append(data)
    
    stacked_data = torch.cat(clusters)

    if shuffle:
        randidxs = torch.randperm(stacked_data.size(0))
        return stacked_data[randidxs]

    return stacked_data


def uniform_normal_gauss_ds(
        n_features:int = 2, 
        n_samples:int = 100,
        n_classes:int = 2,
        std: int = 1,
        device:str = 'cpu',
        shuffle:bool = True) -> torch.Tensor:
    
    upper_means = np.arange(start=1, stop=n_classes//2 + 1).tolist()
    lower_means = -np.arange(start=1, stop=n_classes//2 + 1).tolist()
    stds = [std]*len(means)

    data = normal_gauss_ds(n_features=n_features, n_samples=n_samples, means=means, stds=stds, device=device, shuffle=shuffle)

    return data