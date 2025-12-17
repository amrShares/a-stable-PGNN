import os, sys

from networkx.algorithms.threshold import betweenness_sequence
from sympy.physics.units.systems.si import base_dims

os.chdir(r"C:\Users\asmaa\Desktop\Amr\Master's Thesis v0.4")
sys.path.append(r"C:\Users\asmaa\Desktop\Amr\Master's Thesis v0.4")

import devices
import models
import samplers
import trainers
import utils

import torch
import numpy as np

from optimizers import SOAP

# # parameters
# lambd = 1
#
# # wave number and normalized spacing
# k_0 = 2 * np.pi / lambd
#
# # normalized simulation lengths
# substrate_length = (6 * lambd) * k_0
# core_length = (2 * lambd) * k_0
# cladding_length = (6 * lambd) * k_0
#
# # refractive indices
# n_substrate = 1.44
# n_core = 1.55
# n_cladding = 1.44

# parameters
# lambd = 1
#
# # wave number and normalized spacing
# k_0 = 2 * np.pi / lambd
#
# # normalized simulation lengths
# substrate_length = (6 * lambd) * k_0
# core_length = (2 * lambd) * k_0
# cladding_length = (6 * lambd) * k_0
#
# # refractive indices
# n_substrate = 1.5
# n_core = 1.7
# n_cladding = 1.2

# # parameters
lambd = 1

# wave number and normalized spacing
k_0 = 2 * np.pi / lambd

# normalized simulation lengths
substrate_length = (4 * lambd) * k_0
core_length = (1 * lambd) * k_0
cladding_length = (4 * lambd) * k_0

# refractive indices
n_substrate = 1
n_core = 3
n_cladding = 1

# # study and field types
study = 'TM'
field_type = 'E'

# Preparing simulation parameters
RIs = [n_substrate, n_core, n_cladding]
lengths = [substrate_length, core_length, cladding_length]
reference_device = devices.SlabWaveguide(RIs, lambd, lengths, study=study, field_type=field_type)
memory_type = torch.device('cpu')
dtype = torch.float64

num_modes = 1

save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TM"

EIs_FD = []
modes_errors_FD = []

EIs_DCNN = []
modes_errors_DCNN = []

EIs_FCNN = []
modes_errors_FCNN = []

for i in range(5):
    EIs_DCNN_instance = []
    EIs_FCNN_instance = []
    modes_errors_DCNN_instance = []
    modes_errors_FCNN_instance = []
    print('Discretization No.', i+1)

    for j in range(10):
        dx = (lambd / (i+1) / 10) * k_0

        # FD
        fd_device = devices.SlabWaveguideWithFD(RIs, lambd, lengths, study, field_type, dx)
        _, x, u = fd_device.evaluate(num_modes)
        u = u / np.max(np.abs(u), axis=0, keepdims=True)
        mode_errors_FD = utils.compare_against_analytic(fd_device, x, torch.from_numpy(u), plot=True)

        if j==0:
            EIs_FD.append(_**(1/2))
            modes_errors_FD.append(mode_errors_FD)

        print('EIs FD:', EIs_FD)
        print('Mode Errors FD:', modes_errors_FD)

        # # DCNN
        # # torch.manual_seed(42)
        # base_model = models.discontinuity_capturing_network(1, 64, num_modes, 3, ['DRBF', 'DRBF'])
        # # base_model.apply(models.default_init)
        # # optimzer = torch.optim.Adamax(base_model.parameters(), lr=1e-2)
        # optimizer = SOAP(base_model.parameters(), lr=1e-2, betas = (0.9, 0.99), weight_decay=0, precondition_frequency=10)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1001, 1501], gamma=0.1)
        # device = devices.SlabWaveguideWithPINNs(RIs, lambd, lengths, study, field_type, base_model, True)
        # train_agent = trainers.Trainer(dtype, device, memory_type, [samplers.uniform_grid], optimizer, scheduler, dx, n_lower=None)
        # loss_history, rayleigh_history, mode_errors_list = train_agent.train(2001, num_modes,
        #                                                                      weights=[1, 1e2, 1e1, 1, 1, 1e-1],
        #                                                                      verbose=10, sample_new=False,
        #                                                                      plot_solution=False)
        # num_modes = len(u.T)
        # evaluation_points, eigen_modes_analytic, eigen_funcs_analytic = reference_device.evaluate_analytical(x, num_modes)
        # augmented_x = torch.cat((torch.from_numpy(evaluation_points).reshape(-1, 1), 1/device.refractive_index(torch.from_numpy(evaluation_points).reshape(-1, 1), format='torch')**2), dim=-1).to(memory_type)
        # predictions = device.underlying_model(augmented_x)
        # prediction_normed = predictions.detach().cpu() / torch.max(torch.abs(predictions.detach().cpu()), dim=0, keepdims=True)[0]
        # mode_errors_DCNN = utils.compare_against_analytic(reference_device, x, prediction_normed.detach().cpu(), plot=True)
        # augmented_x.requires_grad_(True)
        # EI_DCNN = []
        # for k in range(num_modes):
        #     EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), k).item() ** (1 / 2))
        # EIs_DCNN_instance.append(EI_DCNN)
        # modes_errors_DCNN_instance.append(mode_errors_DCNN)
        #
        # print('EIs DCNN:', EIs_DCNN_instance)
        # print('modes errors DCNN:', modes_errors_DCNN_instance)
        #
        # FCNN
        # torch.manual_seed(42)
        base_model = models.vanilla_network(1, 64, num_modes, 2, 'DRBF')
        # base_model.apply(models.default_init)
        # optimzer = torch.optim.Adamax(base_model.parameters(), lr=1e-2)
        optimizer = SOAP(base_model.parameters(), lr=1e-2, betas = (0.9, 0.99), weight_decay=0, precondition_frequency=10)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1001, 1501], gamma=0.1)
        device = devices.SlabWaveguideWithPINNs(RIs, lambd, lengths, study, field_type, base_model, False)
        train_agent = trainers.Trainer(dtype, device, memory_type, [samplers.uniform_grid], optimizer, scheduler, dx, n_lower=2.5)
        loss_history, rayleigh_history, mode_errors_list, _ = train_agent.train(2001, num_modes,
                                                                             weights=[1, 1e2, 1e1, 1, 1, 1e-5],
                                                                             verbose=10, sample_new=False,
                                                                             plot_solution=False)
        num_modes = len(u.T)
        evaluation_points, eigen_modes_analytic, eigen_funcs_analytic = reference_device.evaluate_analytical(x, num_modes)
        augmented_x = torch.from_numpy(evaluation_points).reshape(-1, 1).to(memory_type)
        predictions = device.underlying_model(augmented_x)
        prediction_normed = predictions.detach().cpu() / torch.max(torch.abs(predictions.detach().cpu()), dim=0, keepdims=True)[0]
        mode_errors_FCNN = utils.compare_against_analytic(reference_device, x, prediction_normed.detach().cpu(), plot=True)
        augmented_x.requires_grad_(True)
        EI_FCNN = []
        for k in range(num_modes):
            EI_FCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), k).item() ** (1 / 2))
        EIs_FCNN_instance.append(EI_FCNN)
        modes_errors_FCNN_instance.append(mode_errors_FCNN)

        print('EIs FCNN:', EIs_FCNN_instance)
        print('modes errors FCNN:', modes_errors_FCNN_instance)

    EIs_DCNN.append(EIs_DCNN_instance)
    EIs_FCNN.append(EIs_FCNN_instance)
    modes_errors_DCNN.append(modes_errors_DCNN_instance)
    modes_errors_FCNN.append(modes_errors_FCNN_instance)
#
# np.save(save_folder_path + r"\Planar\Several Modes Weakly Guiding\Convergence\EIs_FD.npy", EIs_FD)
# np.save(save_folder_path + r"\Planar\Several Modes Weakly Guiding\Convergence\modes_errors_FD.npy", modes_errors_FD)
#
# np.save(save_folder_path + r"\Planar\Several Modes Weakly Guiding\Convergence\EIs_FCNN.npy", EIs_FCNN)
# np.save(save_folder_path + r"\Planar\Several Modes Weakly Guiding\Convergence\modes_errors_FCNN.npy", modes_errors_FCNN)

# np.save(save_folder_path + r"\Planar\Several Modes Weakly Guiding\Convergence\EIs_DCNN.npy", EIs_DCNN)
# np.save(save_folder_path + r"\Planar\Several Modes Weakly Guiding\Convergence\modes_errors_DCNN.npy", modes_errors_DCNN)
# #
# np.save(save_folder_path + r"\Planar\Several Modes Moderate Contrast\Convergence\EIs_FD.npy", EIs_FD)
# np.save(save_folder_path + r"\Planar\Several Modes Moderate Contrast\Convergence\modes_errors_FD.npy", modes_errors_FD)
#
# np.save(save_folder_path + r"\Planar\Several Modes Moderate Contrast\Convergence\EIs_FCNN.npy", EIs_FCNN)
# np.save(save_folder_path + r"\Planar\Several Modes Moderate Contrast\Convergence\modes_errors_FCNN.npy", modes_errors_FCNN)
#
# np.save(save_folder_path + r"\Planar\Several Modes Moderate Contrast\Convergence\EIs_DCNN.npy", EIs_DCNN)
# np.save(save_folder_path + r"\Planar\Several Modes Moderate Contrast\Convergence\modes_errors_DCNN.npy", modes_errors_DCNN)
#
# np.save(save_folder_path + r"\Planar\Several Modes High Contrast\Convergence\EIs_FD.npy", EIs_FD)
# np.save(save_folder_path + r"\Planar\Several Modes High Contrast\Convergence\modes_errors_FD.npy", modes_errors_FD)
#
np.save(save_folder_path + r"\Planar\Several Modes High Contrast\Convergence\EIs_FCNN.npy", EIs_FCNN)
np.save(save_folder_path + r"\Planar\Several Modes High Contrast\Convergence\modes_errors_FCNN.npy", modes_errors_FCNN)
#
# np.save(save_folder_path + r"\Planar\Several Modes High Contrast\Convergence\EIs_DCNN.npy", EIs_DCNN)
# np.save(save_folder_path + r"\Planar\Several Modes High Contrast\Convergence\modes_errors_DCNN.npy", modes_errors_DCNN)