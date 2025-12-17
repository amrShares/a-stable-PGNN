import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

def compare_against_analytic(device, x, u, plot=False, verbose=False):
    mode_errors = []
    num_samples, num_modes = len(u), len(u.T)

    evaluation_points, eigen_modes_analytic, eigen_funcs_analytic = device.evaluate_analytical(x, num_modes)
    
    if device.study == 'TM' and device.field_type=='E':
      print('adjusting analytic solution for Ex in a TM study')
      eigen_funcs_analytic = eigen_funcs_analytic / device.refractive_index(evaluation_points.reshape(-1), format='numpy')**2
    
    eigen_funcs_analytic /= np.max(np.abs(eigen_funcs_analytic), axis=-1, keepdims=True)
    eigen_funcs_analytic = eigen_funcs_analytic.reshape(num_modes, num_samples)
    if verbose:
        print('*'*10)
        print('mode errors')
    for i in range(num_modes):
        errs1 = np.inf
        errs2 = np.inf
        if verbose:
            print('*' * 10)
        for j in range(num_modes):
            prediction_normed = u[:, i].detach().cpu() / torch.max(torch.abs(u[:, i]).cpu())
            assert prediction_normed.shape == eigen_funcs_analytic[i].shape
            err_1 = (torch.mean((prediction_normed - eigen_funcs_analytic[j]) ** 2) / np.mean(
                eigen_funcs_analytic[j] ** 2)) ** 0.5 * 100
            err_2 = (torch.mean((prediction_normed + eigen_funcs_analytic[j]) ** 2) / np.mean(
                eigen_funcs_analytic[j] ** 2)) ** 0.5 * 100
            errs1 = min(abs(errs1), abs(err_1))
            errs2 = min(abs(errs2), (err_2))
        mode_errors.append(min(abs(errs1.item()), abs(errs2.item())))
        if verbose:
            print(f'relative error for mode {i} is {round(mode_errors[-1], 5)}%')
            print('*' * 10)
    prediction_normed = u.detach().cpu() / torch.max(torch.abs(u).cpu(), dim=0, keepdim=True)[0]
    if plot:
        for i in range(num_modes):
            plt.plot(eigen_funcs_analytic[i], label='analytic')
            plt.plot(prediction_normed[:, i], linestyle='--', label='prediction')
            plt.legend()
            plt.show()
    return mode_errors

def chebyshev_nodes_first_kind(n: int, device=None, dtype=None):
    k = torch.arange(1, n + 1, device=device, dtype=dtype)
    return torch.cos((2 * k - 1) * math.pi / (2 * n))

class InvertedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, error):
        abs_error = torch.abs(error)
        quadratic = (0.5 / self.delta) * error ** 2  - self.delta/2
        return torch.where(abs_error < self.delta, abs_error, quadratic).mean()

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, error):
        abs_error = torch.abs(error)
        quadratic = 0.5 * error ** 2
        linear = self.delta * (torch.abs(error) - 0.5 * self.delta)
        return torch.where(abs_error < self.delta, quadratic, linear)