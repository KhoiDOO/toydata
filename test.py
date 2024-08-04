from toydata import normal_gauss_ds, uniform_normal_gauss_ds

def test_normal_gauss_ds():
    print('Normal Gaussian Dataset Testing')
    ds=normal_gauss_ds(n_features=2, n_samples=100, means=[-0.5, 0.5], stds=[1, 1], device='cpu', shuffle=True)

    print(f'Data Shape: {ds.shape}')
    print(f'Class 0 Mean: {ds[ds[:, -1] == 0].mean(0)}')
    print(f'Class 1 Mean: {ds[ds[:, -1] == 1].mean(0)}')
    print(f'Catefory samples: {ds[:10, -1]}')

def test_uniform_normal_gauss_ds():
    print('Uniform Normal Gaussian Dataset Testing')
    ds=uniform_normal_gauss_ds(n_features=2, n_samples=100, n_classes=2, std=1, device='cpu', shuffle=True)

    print(f'Data Shape: {ds.shape}')
    print(f'Class 0 Mean: {ds[ds[:, -1] == 0].mean(0)}')
    print(f'Class 1 Mean: {ds[ds[:, -1] == 1].mean(0)}')
    print(f'Catefory samples: {ds[:10, -1]}')


if __name__ == '__main__':
    test_normal_gauss_ds()
    test_uniform_normal_gauss_ds()