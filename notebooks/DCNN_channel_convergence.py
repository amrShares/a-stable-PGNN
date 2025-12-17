import os, sys
from sched import scheduler

os.chdir(r"C:\Users\asmaa\Desktop\Amr\Master's Thesis v0.4")
sys.path.append(r"C:\Users\asmaa\Desktop\Amr\Master's Thesis v0.4")

import devices
import models
import samplers
import trainers
import utils
import optimizers

import torch
import numpy as np

# parameters
lambd = 1.55

# wave number and normalized spacing
k_0 = 2 * np.pi / lambd

# model hyperparameters
m = 2

# simulation device parameters
memory_type = torch.device('cuda')
dtype = torch.float32

weights = [1, 1, 1, 1, 1, 1e-1]
m = 1e1

# # number of modes
num_modes = 1
#########################################################################
# # device 1
# # simulation lengths
core_length_x = 2 * k_0 # W
core_length_y = 1 * k_0 # H

substrate_length_x = 3.0 * k_0 # Xs
substrate_length_y = 4.0 * k_0 # Ys

# refractive indices
n_core = 1.55
n_substrate = 1.44

##############################################################################

# Preparing simulation parameters
lengths_x = [substrate_length_x, core_length_x, substrate_length_x]
lengths_y = [substrate_length_y, core_length_y, substrate_length_y]
RIs = [n_substrate, n_core]


# study and field types
study = 'TE'
field_type = 'E'


reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')

save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TE"

EIs_FD = []
modes_errors_FD = []

EIs_DCNN = []
modes_errors_DCNN = []

for i in range(5):
    EIs_DCNN_instance = []
    print('Discretization No.', i+1)
    for j in range(10):
        dx = k_0 / ((i+1)*5) # For device 1
        dy = k_0 / ((i+1)*10) # For device 1

        # FD
        fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
        RI_squared, x, u = fd_device.evaluate()
        u = u / np.max(np.abs(u), axis=0, keepdims=True)
        if j==0:
            EIs_FD.append(RI_squared.item()**(1/2))

        print(EIs_FD)

        # DCNN
        # torch.manual_seed(42)
        base_model = models.discontinuity_capturing_network(2, 64, num_modes, 3, ['DRBF', 'DRBF'])
        base_model.apply(models.default_init)
        # base_model = models.DCNN(2, 64, num_modes, 3, n_core, ['DRBF', 'DRBF'])
        # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 8e-3)
        # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 8e-3)
        optimizer = optimizers.SOAP(
            base_model.parameters(),
            lr=1e-2,
            betas=(0.9, 0.99),
            weight_decay=0,
            precondition_frequency=1
        )
        # optimizer = optimizers.SOAP(
        #     [
        #         {'params': [base_model.eigenvalue], 'lr': 1e-3},
        #         {'params': [p for name, p in base_model.named_parameters() if name != 'eigenvalue'], 'lr': 1e-2}
        #     ],
        #     lr=1e-2,
        #     betas=(0.9, 0.99),
        #     weight_decay=0,
        #     precondition_frequency=1
        # )
        # optimizer = torch.optim.SGD(base_model.parameters(), lr=1e-2, momentum=0.9)
        # optimizer = optimizers.LevenbergMarquardt(base_model.parameters(), lr=8e-3, betas=(0.95, 0.95), weight_decay=0, precondition_frequency=1)
        # optimizer = optimizers.SOAP(base_model.parameters(), lr=8e-3, betas=(0.9, 0.9), weight_decay=0, precondition_frequency=1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2001], gamma=1e-1)

        device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
        train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
                                         optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=None, n_upper=None)

        loss_history, rayleigh_history, mode_errors_list = train_agent.train(2001, num_modes, weights=weights, m=m,
                                                                             verbose=10, calc_error=False,
                                                                             plot_solution= True, save_checkpoint=False)
        num_modes = len(u.T)
        device.underlying_model.to(dtype)
        x = torch.from_numpy(x)
        augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
        u = device.underlying_model(augmented_x)
        u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
        augmented_x.requires_grad_(True)
        EI_DCNN = []
        for k in range(num_modes):
            EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), k).item() ** (1 / 2))
            # EI_DCNN.append(np.sqrt(np.max(np.array(rayleigh_history[0][-3:]))))
        EIs_DCNN_instance.append(EI_DCNN)

        print(EIs_DCNN_instance)

    EIs_DCNN.append(EIs_DCNN_instance)

np.save(save_folder_path + r"\NonPlanar\Single Mode Weakly Guiding\Convergence\EIs_DCNN.npy", EIs_DCNN)

# ######################################################################
#
# # study and field types
# study = 'TM'
#
# # Preparing simulation parameters
# reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')
#
# save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TM"
#
# EIs_FD = []
# modes_errors_FD = []
#
# EIs_DCNN = []
# modes_errors_DCNN = []
#
# for i in range(5):
#     EIs_DCNN_instance = []
#     print('Discretization No.', i+1)
#     for j in range(10):
#         dx = k_0 / ((i+1)*5) # For device 1
#         dy = k_0 / ((i+1)*10) # For device 1
#
#         # FD
#         fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
#         RI_squared, x, u = fd_device.evaluate()
#         u = u / np.max(np.abs(u), axis=0, keepdims=True)
#         if j==0:
#             EIs_FD.append(RI_squared.item()**(1/2))
#
#         print(EIs_FD)
#
#         # DCNN
#         # torch.manual_seed(42)
#         base_model = models.discontinuity_capturing_network(2, 64, num_modes, 3, ['DRBF', 'DRBF'])
#         base_model.apply(models.default_init)
#         # base_model = models.DCNN(2, 64, num_modes, 3, n_core, ['DRBF', 'DRBF'])
#         # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 8e-3)
#         # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 8e-3)
#         optimizer = optimizers.SOAP(
#             base_model.parameters(),
#             lr=1e-2,
#             betas=(0.9, 0.99),
#             weight_decay=0,
#             precondition_frequency=1
#         )
#         # optimizer = optimizers.SOAP(
#         #     [
#         #         {'params': [base_model.eigenvalue], 'lr': 1e-3},
#         #         {'params': [p for name, p in base_model.named_parameters() if name != 'eigenvalue'], 'lr': 1e-2}
#         #     ],
#         #     lr=1e-2,
#         #     betas=(0.9, 0.99),
#         #     weight_decay=0,
#         #     precondition_frequency=1
#         # )
#         # optimizer = torch.optim.SGD(base_model.parameters(), lr=1e-2, momentum=0.9)
#         # optimizer = optimizers.LevenbergMarquardt(base_model.parameters(), lr=8e-3, betas=(0.95, 0.95), weight_decay=0, precondition_frequency=1)
#         # optimizer = optimizers.SOAP(base_model.parameters(), lr=8e-3, betas=(0.9, 0.9), weight_decay=0, precondition_frequency=1)
#         # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1001)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10001], gamma=1e-1)
#
#         device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
#         train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
#                                          optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=None, n_upper=None)
#
#         loss_history, rayleigh_history, mode_errors_list = train_agent.train(10001, num_modes, weights=weights, m=m,
#                                                                              verbose=10, calc_error=False,
#                                                                              plot_solution= True, save_checkpoint=False)
#
#         num_modes = len(u.T)
#         device.underlying_model.to(dtype)
#         x = torch.from_numpy(x)
#         augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
#         u = device.underlying_model(augmented_x)
#         u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
#         augmented_x.requires_grad_(True)
#         EI_DCNN = []
#         for k in range(num_modes):
#             EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), k).item() ** (1 / 2))
#             # EI_DCNN.append(np.sqrt(np.max(np.array(rayleigh_history[0][-3:]))))
#         EIs_DCNN_instance.append(EI_DCNN)
#
#         print(EIs_DCNN_instance)
#
#     EIs_DCNN.append(EIs_DCNN_instance)
#
#
# np.save(save_folder_path + r"\NonPlanar\Single Mode Weakly Guiding\Convergence\EIs_DCNN.npy", EIs_DCNN)

#############################################################################

# # device 2
# simulation lengths
#
# core_length_x = 2 * k_0 # W
# core_length_y = 2 * k_0 # H
#
# substrate_length_x = 2 * k_0 # Xs
# substrate_length_y = 2 * k_0 # Ys
#
# # refractive indices
# n_core = 1.7
# n_substrate = 1.2
#
# # #############################################################################
#
# # study and field types
# study = 'TE'
# field_type = 'E'
#
# # Preparing simulation parameters
# lengths_x = [substrate_length_x, core_length_x, substrate_length_x]
# lengths_y = [substrate_length_y, core_length_y, substrate_length_y]
# RIs = [n_substrate, n_core]
# reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')
#
# save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TE"
#
# EIs_FD = []
# modes_errors_FD = []
#
# EIs_DCNN = []
# modes_errors_DCNN = []
#
# for i in range(5):
#     EIs_DCNN_instance = []
#     print('Discretization No.', i+1)
#     for j in range(10):
#         dx = k_0 / ((2*i+1)*5) # For device 1
#         dy = k_0 / ((2*i+1)*5) # For device 1
#
#         # FD
#         fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
#         RI_squared, x, u = fd_device.evaluate()
#         u = u / np.max(np.abs(u), axis=0, keepdims=True)
#         if j==0:
#             EIs_FD.append(RI_squared.item()**(1/2))
#
#         print(EIs_FD)
#
#         # DCNN
#         # torch.manual_seed(42)
#         base_model = models.discontinuity_capturing_network(2, 64, num_modes, 3, ['DRBF', 'DRBF'])
#         base_model.apply(models.default_init)
#         # base_model = models.DCNN(2, 64, num_modes, 3, n_core, ['DRBF', 'DRBF'])
#         # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 8e-3)
#         # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 8e-3)
#         optimizer = optimizers.SOAP(
#             base_model.parameters(),
#             lr=1e-2,
#             betas=(0.9, 0.99),
#             weight_decay=0,
#             precondition_frequency=1
#         )
#         # optimizer = optimizers.SOAP(
#         #     [
#         #         {'params': [base_model.eigenvalue], 'lr': 1e-3},
#         #         {'params': [p for name, p in base_model.named_parameters() if name != 'eigenvalue'], 'lr': 1e-2}
#         #     ],
#         #     lr=1e-2,
#         #     betas=(0.9, 0.99),
#         #     weight_decay=0,
#         #     precondition_frequency=1
#         # )
#         # optimizer = torch.optim.SGD(base_model.parameters(), lr=1e-2, momentum=0.9)
#         # optimizer = optimizers.LevenbergMarquardt(base_model.parameters(), lr=8e-3, betas=(0.95, 0.95), weight_decay=0, precondition_frequency=1)
#         # optimizer = optimizers.SOAP(base_model.parameters(), lr=8e-3, betas=(0.9, 0.9), weight_decay=0, precondition_frequency=1)
#         # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1001)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10001], gamma=1e-1)
#
#         device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
#         train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
#                                          optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=1.55, n_upper=None)
#
#         loss_history, rayleigh_history, mode_errors_list = train_agent.train(10001, num_modes, weights=weights, m=m,
#                                                                              verbose=10, calc_error=False,
#                                                                              plot_solution= True, save_checkpoint=False)
#
#         num_modes = len(u.T)
#         device.underlying_model.to(dtype)
#         x = torch.from_numpy(x)
#         augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
#         u = device.underlying_model(augmented_x)
#         u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
#         augmented_x.requires_grad_(True)
#         EI_DCNN = []
#         for k in range(num_modes):
#             EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), k).item() ** (1 / 2))
#             # EI_DCNN.append(np.sqrt(np.max(np.array(rayleigh_history[0][-3:]))))
#         EIs_DCNN_instance.append(EI_DCNN)
#
#         print(EIs_DCNN_instance)
#
#     EIs_DCNN.append(EIs_DCNN_instance)
#
# np.save(save_folder_path + r"\NonPlanar\Single Mode Moderate Contrast\Convergence\EIs_DCNN.npy", EIs_DCNN)

##########################################################################
#
# # study and field types
# study = 'TM'
#
# # Preparing simulation parameters
# reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')
#
# save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TM"
#
# EIs_FD = []
# modes_errors_FD = []
#
# EIs_DCNN = []
# modes_errors_DCNN = []
#
# for i in range(5):
#     # dx = k_0 / ((2*i+1)*10) # For device 3
#     # dy = k_0 / ((2*i+1)*10) # For device 3
#     dx = k_0 / ((i+1)*20) # For device 2
#     dy = k_0 / ((i+1)*20) # For device 2
#
#     # FD
#     fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
#     RI_squared, x, u = fd_device.evaluate()
#     u = u / np.max(np.abs(u), axis=0, keepdims=True)
#     EIs_FD.append(RI_squared.item()**(1/2))
#
#     print(EIs_FD)
#
#     # DCNN
#     torch.manual_seed(42)
#     base_model = models.discontinuity_capturing_network(2, 64, num_modes, 2, 'Siren')
#     optimizer = torch.optim.Adamax(base_model.parameters(), lr=8e-3)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20000], gamma=1e0)
#     device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
#     train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
#                                      optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=None, n_upper=None)
#
#     loss_history, rayleigh_history, mode_errors_list = train_agent.train(5001, num_modes, weights=weights, m=m,
#                                                                          verbose=1, calc_error=False,
#                                                                          plot_solution= True, sample_new=False)
#     num_modes = len(u.T)
#     device.underlying_model.to(dtype)
#     x = torch.from_numpy(x)
#     augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
#     u = device.underlying_model(augmented_x)
#     u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
#     augmented_x.requires_grad_(True)
#     EI_DCNN = []
#     for i in range(num_modes):
#         EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), i).item() ** (1 / 2))
#     EIs_DCNN.append(EI_DCNN)
#
#     print(EIs_DCNN)
#
# np.save(save_folder_path + r"\NonPlanar\Single Mode Moderate Contrast\Convergence\EIs_DCNN.npy", EIs_DCNN)

# #####################################################################################
#
# device 3
# simulation lengths
core_length_x = 1 * k_0 # W
core_length_y = 2 * k_0 # H

substrate_length_x = 3 * k_0 # Xs
substrate_length_y = 2 * k_0 # Ys

# refractive indices
n_core = 3.0
n_substrate = 1.0

# # study and field types
study = 'TE'
field_type = 'E'
#
# # Preparing simulation parameters
lengths_x = [substrate_length_x, core_length_x, substrate_length_x]
lengths_y = [substrate_length_y, core_length_y, substrate_length_y]
RIs = [n_substrate, n_core]
reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')

# save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TE"
#
# EIs_FD = []
# modes_errors_FD = []
#
# EIs_DCNN = []
# modes_errors_DCNN = []
#
# for i in range(5):
#     EIs_DCNN_instance = []
#     print('Discretization No.', i+1)
#     for j in range(10):
#         dx = k_0 / ((2*i+1)*5) # For device 1
#         dy = k_0 / ((2*i+1)*5) # For device 1
#
#         # FD
#         fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
#         RI_squared, x, u = fd_device.evaluate()
#         u = u / np.max(np.abs(u), axis=0, keepdims=True)
#         if j==0:
#             EIs_FD.append(RI_squared.item()**(1/2))
#
#         print(EIs_FD)
#
#         # DCNN
#         # torch.manual_seed(42)
#         base_model = models.discontinuity_capturing_network(2, 64, num_modes, 3, ['DRBF', 'DRBF'])
#         base_model.apply(models.default_init)
#         # base_model = models.DCNN(2, 64, num_modes, 2, n_core, ['RBF', 'RBF'])
#         # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 8e-3)
#         # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 8e-3)
#         optimizer = optimizers.SOAP(
#             base_model.parameters(),
#             lr=1e-2,
#             betas=(0.9, 0.99),
#             weight_decay=0,
#             precondition_frequency=1
#         )
#         # optimizer = optimizers.SOAP(
#         #     [
#         #         {'params': [base_model.eigenvalue], 'lr': 3e-4},
#         #         {'params': [p for name, p in base_model.named_parameters() if name != 'eigenvalue'], 'lr': 8e-3}
#         #     ],
#         #     lr=1e-2,
#         #     betas=(0.9, 0.99),
#         #     weight_decay=0,
#         #     precondition_frequency=1
#         # )
#         # optimizer = torch.optim.SGD(base_model.parameters(), lr=1e-2, momentum=0.9)
#         # optimizer = optimizers.LevenbergMarquardt(base_model.parameters(), lr=8e-3, betas=(0.95, 0.95), weight_decay=0, precondition_frequency=1)
#         # optimizer = optimizers.SOAP(base_model.parameters(), lr=8e-3, betas=(0.9, 0.9), weight_decay=0, precondition_frequency=1)
#         # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1001)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2001], gamma=1e-1)
#
#         device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
#         train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
#                                          optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=2.5, n_upper=None)
#
#         loss_history, rayleigh_history, mode_errors_list = train_agent.train(2001, num_modes, weights=weights, m=m,
#                                                                              verbose=10, calc_error=False,
#                                                                              plot_solution= True, save_checkpoint=False)
#
#         num_modes = len(u.T)
#         device.underlying_model.to(dtype)
#         x = torch.from_numpy(x)
#         augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
#         u = device.underlying_model(augmented_x)
#         u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
#         augmented_x.requires_grad_(True)
#         EI_DCNN = []
#         for k in range(num_modes):
#             EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), k).item() ** (1 / 2))
#             # EI_DCNN.append(np.sqrt(np.max(np.array(rayleigh_history[0][-3:]))))
#
#
#
#         EIs_DCNN_instance.append(EI_DCNN)
#
#         print(EIs_DCNN_instance)
#
#     EIs_DCNN.append(EIs_DCNN_instance)
#
# np.save(save_folder_path + r"\NonPlanar\Single Mode High Contrast\Convergence\EIs_DCNN.npy", EIs_DCNN)
#
###############################################################################



# study and field types
study = 'TM'

# Preparing simulation parameters
reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')

save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TM"

EIs_FD = []
modes_errors_FD = []

EIs_DCNN = []
modes_errors_DCNN = []

for i in range(5):
    EIs_DCNN_instance = []
    print('Discretization No.', i+1)
    for j in range(10):
        dx = k_0 / ((2*i+1)*5) # For device 1
        dy = k_0 / ((2*i+1)*5) # For device 1

        # FD
        fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
        RI_squared, x, u = fd_device.evaluate()
        u = u / np.max(np.abs(u), axis=0, keepdims=True)
        if j==0:
            EIs_FD.append(RI_squared.item()**(1/2))

        print(EIs_FD)

        # DCNN
        # torch.manual_seed(42)
        base_model = models.discontinuity_capturing_network(2, 64, num_modes, 3, ['DRBF', 'DRBF'])
        base_model.apply(models.default_init)
        # base_model = models.DCNN(2, 64, num_modes, 3, n_core, ['DRBF', 'DRBF'])
        # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 8e-3)
        # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 8e-3)
        optimizer = optimizers.SOAP(
            base_model.parameters(),
            lr=1e-2,
            betas=(0.9, 0.99),
            weight_decay=0,
            precondition_frequency=1
        )
        # optimizer = optimizers.SOAP(
        #     [
        #         {'params': [base_model.eigenvalue], 'lr': 1e-3},
        #         {'params': [p for name, p in base_model.named_parameters() if name != 'eigenvalue'], 'lr': 1e-2}
        #     ],
        #     lr=1e-2,
        #     betas=(0.9, 0.99),
        #     weight_decay=0,
        #     precondition_frequency=1
        # )
        # optimizer = torch.optim.SGD(base_model.parameters(), lr=1e-2, momentum=0.9)
        # optimizer = optimizers.LevenbergMarquardt(base_model.parameters(), lr=8e-3, betas=(0.95, 0.95), weight_decay=0, precondition_frequency=1)
        # optimizer = optimizers.SOAP(base_model.parameters(), lr=8e-3, betas=(0.9, 0.9), weight_decay=0, precondition_frequency=1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2001], gamma=1e-1)

        device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
        train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
                                         optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=2.5, n_upper=None)

        loss_history, rayleigh_history, mode_errors_list = train_agent.train(2001, num_modes, weights=weights, m=m,
                                                                             verbose=10, calc_error=False,
                                                                             plot_solution= True, save_checkpoint=False)

        num_modes = len(u.T)
        device.underlying_model.to(dtype)
        x = torch.from_numpy(x)
        augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
        u = device.underlying_model(augmented_x)
        u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
        augmented_x.requires_grad_(True)
        EI_DCNN = []
        for k in range(num_modes):
            EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), k).item() ** (1 / 2))
            # EI_DCNN.append(np.sqrt(np.max(np.array(rayleigh_history[0][-3:]))))

        EIs_DCNN_instance.append(EI_DCNN)

        print(EIs_DCNN_instance)

    EIs_DCNN.append(EIs_DCNN_instance)

np.save(save_folder_path + r"\NonPlanar\Single Mode High Contrast\Convergence\EIs_DCNN.npy", EIs_DCNN)

####################################################################################

