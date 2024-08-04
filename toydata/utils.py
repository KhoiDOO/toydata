import numpy as np


def check_means_stds_1d(means:list[float], stds:list[float]):
    if len(means) != len(stds):
        raise ValueError(f'len of means {len(means)} and stds {len(stds)} need to be equal')

def check_device(device:str):
    if device not in ['cpu', 'cuda']:
        raise ValueError(f"device need to be in ['cpu', 'device']")

def check_n_feature(n_features:int):
    if n_features <= 0:
        raise ValueError(f'n_features {n_features} need to be larger than 0')

def circle(n_features:int=2, n_classes:int=2, radius:float=1):
    single_thetas = np.arange(0, 360, 360/n_classes)
    single_radians = np.radians(single_thetas)

    radians = np.concatenate()
    
    xs = radius * np.cos(radians)
    ys = radius * np.sin(radians)