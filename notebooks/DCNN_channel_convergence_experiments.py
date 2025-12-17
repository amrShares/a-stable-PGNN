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
dtype = torch.float64

# number of modes
num_modes = 1




############################################################################
# device 1
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

save_folder_path = os.getcwd()

for j in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
    # Define the folder path (use raw string or double backslashes for Windows paths)
    folder_path = r"TE\NonPlanar\Single Mode Weakly Guiding\Convergence"
    # Create the directory and all parent directories if they don't exist
    os.makedirs(os.path.join(os.getcwd(), folder_path, f'{j}'), exist_ok=True)

    weights = [1, 1, 1, 1, 1, j]

    EIs_FD = []
    modes_errors_FD = []

    EIs_DCNN = []
    modes_errors_DCNN = []

    for i in range(1):
        dx = k_0 / ((i+1)*5) # For device 1
        dy = k_0 / ((i+1)*10) # For device 1

        # FD
        fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
        RI_squared, x, u = fd_device.evaluate()
        u = u / np.max(np.abs(u), axis=0, keepdims=True)
        EIs_FD.append(RI_squared.item()**(1/2))

        print(EIs_FD)

        # DCNN
        torch.manual_seed(4321)
        base_model = models.discontinuity_capturing_network(2, 128, num_modes, 3, ['RBF', 'RBF'])
        # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 1e-3)
        optimizer = optimizers.SOAP(base_model.parameters(), lr=8e-3, betas=(0.9, 0.9), precondition_frequency=1, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10001)
        device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
        train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
                                         optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=None, n_upper=None)

        loss_history, rayleigh_history, mode_errors_list = train_agent.train(10001, num_modes, weights=weights, m=m,
                                                                             verbose=1, calc_error=False,
                                                                             plot_solution= True, save_checkpoint=True)
        num_modes = len(u.T)
        device.underlying_model.to(dtype)
        x = torch.from_numpy(x)
        augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
        u = device.underlying_model(augmented_x)
        u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
        augmented_x.requires_grad_(True)
        EI_DCNN = []
        for i in range(num_modes):
            EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), i).item() ** (1 / 2))
        EIs_DCNN.append(EI_DCNN)

        print(EIs_DCNN)


    np.save(os.path.join(save_folder_path, folder_path, f'{j}', 'EIs_DCNN.npy'), EIs_DCNN)

###########################################################################

# study and field types
study = 'TM'

# Preparing simulation parameters
reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')

for j in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
    # Define the folder path (use raw string or double backslashes for Windows paths)
    folder_path = r"TM\NonPlanar\Single Mode Weakly Guiding\Convergence"
    # Create the directory and all parent directories if they don't exist
    os.makedirs(os.path.join(os.getcwd(), folder_path, f'{j}'), exist_ok=True)

    weights = [1, 1, 1, 1, 1, j]

    EIs_FD = []
    modes_errors_FD = []

    EIs_DCNN = []
    modes_errors_DCNN = []

    for i in range(1):
        dx = k_0 / ((i+1)*5) # For device 1
        dy = k_0 / ((i+1)*10) # For device 1

        # FD
        fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
        RI_squared, x, u = fd_device.evaluate()
        u = u / np.max(np.abs(u), axis=0, keepdims=True)
        EIs_FD.append(RI_squared.item()**(1/2))

        print(EIs_FD)

        # DCNN
        torch.manual_seed(4321)
        base_model = models.discontinuity_capturing_network(2, 128, num_modes, 3, ['RBF', 'RBF'])
        # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 1e-3)
        optimizer = optimizers.SOAP(base_model.parameters(), lr=8e-3, betas=(0.9, 0.9), precondition_frequency=1, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10001)
        device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
        train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
                                         optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=None, n_upper=None)

        loss_history, rayleigh_history, mode_errors_list = train_agent.train(10001, num_modes, weights=weights, m=m,
                                                                             verbose=1, calc_error=False,
                                                                             plot_solution= True, save_checkpoint=True)
        num_modes = len(u.T)
        device.underlying_model.to(dtype)
        x = torch.from_numpy(x)
        augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
        u = device.underlying_model(augmented_x)
        u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
        augmented_x.requires_grad_(True)
        EI_DCNN = []
        for i in range(num_modes):
            EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), i).item() ** (1 / 2))
        EIs_DCNN.append(EI_DCNN)

        print(EIs_DCNN)

    np.save(os.path.join(save_folder_path, folder_path, f'{j}', 'EIs_DCNN.npy'), EIs_DCNN)

##############################################################################

# device 2
# simulation lengths

core_length_x = 2 * k_0 # W
core_length_y = 2 * k_0 # H

substrate_length_x = 2 * k_0 # Xs
substrate_length_y = 2 * k_0 # Ys

# refractive indices
n_core = 1.7
n_substrate = 1.2

# weights = [1, 1, 2, 1, 1, 1/2]
# m = n_core**2
#############################################################################

# study and field types
study = 'TE'
field_type = 'E'

# Preparing simulation parameters
lengths_x = [substrate_length_x, core_length_x, substrate_length_x]
lengths_y = [substrate_length_y, core_length_y, substrate_length_y]
RIs = [n_substrate, n_core]
reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')

for j in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
    # Define the folder path (use raw string or double backslashes for Windows paths)
    folder_path = r"TE\NonPlanar\Single Mode Moderate Contrast\Convergence"
    # Create the directory and all parent directories if they don't exist
    os.makedirs(os.path.join(os.getcwd(), folder_path, f'{j}'), exist_ok=True)

    weights = [1, 1, 1, 1, 1, j]

    EIs_FD = []
    modes_errors_FD = []

    EIs_DCNN = []
    modes_errors_DCNN = []

    for i in range(1):
        dx = k_0 / ((i+1)*5) # For device 1
        dy = k_0 / ((i+1)*10) # For device 1

        # FD
        fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
        RI_squared, x, u = fd_device.evaluate()
        u = u / np.max(np.abs(u), axis=0, keepdims=True)
        EIs_FD.append(RI_squared.item()**(1/2))

        print(EIs_FD)

        # DCNN
        torch.manual_seed(4321)
        base_model = models.discontinuity_capturing_network(2, 128, num_modes, 3, ['RBF', 'RBF'])
        # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 1e-3)
        optimizer = optimizers.SOAP(base_model.parameters(), lr=8e-3, betas=(0.9, 0.9), precondition_frequency=1, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10001)
        device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
        train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
                                         optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=None, n_upper=None)

        loss_history, rayleigh_history, mode_errors_list = train_agent.train(10001, num_modes, weights=weights, m=m,
                                                                             verbose=1, calc_error=False,
                                                                             plot_solution= True, save_checkpoint=True)
        num_modes = len(u.T)
        device.underlying_model.to(dtype)
        x = torch.from_numpy(x)
        augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
        u = device.underlying_model(augmented_x)
        u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
        augmented_x.requires_grad_(True)
        EI_DCNN = []
        for i in range(num_modes):
            EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), i).item() ** (1 / 2))
        EIs_DCNN.append(EI_DCNN)

        print(EIs_DCNN)

    np.save(os.path.join(save_folder_path, folder_path, f'{j}', 'EIs_DCNN.npy'), EIs_DCNN)

#############################################################################
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
n_core = 3
n_substrate = 1
#
# weights = [1, 1, n_core**2, 1, 1, 1/10]
# m = n_core**2
# # study and field types
study = 'TE'
field_type = 'E'
#
# # Preparing simulation parameters
lengths_x = [substrate_length_x, core_length_x, substrate_length_x]
lengths_y = [substrate_length_y, core_length_y, substrate_length_y]
RIs = [n_substrate, n_core]
reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')

for j in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
    # Define the folder path (use raw string or double backslashes for Windows paths)
    folder_path = r"TE\NonPlanar\Single Mode High Contrast\Convergence"
    # Create the directory and all parent directories if they don't exist
    os.makedirs(os.path.join(os.getcwd(), folder_path, f'{j}'), exist_ok=True)

    weights = [1, 1, 1, 1, 1, j]

    EIs_FD = []
    modes_errors_FD = []

    EIs_DCNN = []
    modes_errors_DCNN = []

    for i in range(1):
        dx = k_0 / ((i+1)*5) # For device 1
        dy = k_0 / ((i+1)*10) # For device 1

        # FD
        fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
        RI_squared, x, u = fd_device.evaluate()
        u = u / np.max(np.abs(u), axis=0, keepdims=True)
        EIs_FD.append(RI_squared.item()**(1/2))

        print(EIs_FD)

        # DCNN
        torch.manual_seed(4321)
        base_model = models.discontinuity_capturing_network(2, 128, num_modes, 3, ['RBF', 'RBF'])
        # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 1e-3)
        optimizer = optimizers.SOAP(base_model.parameters(), lr=8e-3, betas=(0.9, 0.9), precondition_frequency=1, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10001)
        device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
        train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
                                         optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=None, n_upper=None)

        loss_history, rayleigh_history, mode_errors_list = train_agent.train(10001, num_modes, weights=weights, m=m,
                                                                             verbose=1, calc_error=False,
                                                                             plot_solution= True, save_checkpoint=True)
        num_modes = len(u.T)
        device.underlying_model.to(dtype)
        x = torch.from_numpy(x)
        augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
        u = device.underlying_model(augmented_x)
        u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
        augmented_x.requires_grad_(True)
        EI_DCNN = []
        for i in range(num_modes):
            EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), i).item() ** (1 / 2))
        EIs_DCNN.append(EI_DCNN)

        print(EIs_DCNN)

    np.save(os.path.join(save_folder_path, folder_path, f'{j}', 'EIs_DCNN.npy'), EIs_DCNN)

###############################################################################



# study and field types
study = 'TM'

# Preparing simulation parameters
reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')

for j in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
    # Define the folder path (use raw string or double backslashes for Windows paths)
    folder_path = r"TM\NonPlanar\Single Mode High Contrast\Convergence"
    # Create the directory and all parent directories if they don't exist
    os.makedirs(os.path.join(os.getcwd(), folder_path, f'{j}'), exist_ok=True)

    weights = [1, 1, 1, 1, 1, j]

    EIs_FD = []
    modes_errors_FD = []

    EIs_DCNN = []
    modes_errors_DCNN = []

    for i in range(1):
        dx = k_0 / ((i+1)*5) # For device 1
        dy = k_0 / ((i+1)*10) # For device 1

        # FD
        fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
        RI_squared, x, u = fd_device.evaluate()
        u = u / np.max(np.abs(u), axis=0, keepdims=True)
        EIs_FD.append(RI_squared.item()**(1/2))

        print(EIs_FD)

        # DCNN
        torch.manual_seed(4321)
        base_model = models.discontinuity_capturing_network(2, 128, num_modes, 3, ['RBF', 'RBF'])
        # optimizer = torch.optim.Adamax(base_model.parameters(), lr = 1e-3)
        optimizer = optimizers.SOAP(base_model.parameters(), lr=8e-3, betas=(0.9, 0.9), precondition_frequency=1, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10001)
        device = devices.BuriedChannelWaveguideWithPINNs(RIs, lambd, lengths_x, lengths_y, study, 'E', base_model, 1, True)
        train_agent = trainers.Trainer2D(dtype, device, fd_device, memory_type, samplers.uniform_grid_2d,
                                         optimizer, scheduler, dx=dx, dy=dy, num_samples=None, n_lower=None, n_upper=None)

        loss_history, rayleigh_history, mode_errors_list = train_agent.train(10001, num_modes, weights=weights, m=m,
                                                                             verbose=1, calc_error=False,
                                                                             plot_solution= True, save_checkpoint=True)
        num_modes = len(u.T)
        device.underlying_model.to(dtype)
        x = torch.from_numpy(x)
        augmented_x = torch.cat((x, device.make_features(x)), dim=-1).to(memory_type).to(dtype)
        u = device.underlying_model(augmented_x)
        u = u.detach().cpu() / torch.max(torch.abs(u.detach().cpu()), dim=0, keepdims=True)[0]
        augmented_x.requires_grad_(True)
        EI_DCNN = []
        for i in range(num_modes):
            EI_DCNN.append(device.calculate_eigen_value(augmented_x.to(memory_type), i).item() ** (1 / 2))
        EIs_DCNN.append(EI_DCNN)

        print(EIs_DCNN)

    np.save(os.path.join(save_folder_path, folder_path, f'{j}', 'EIs_DCNN.npy'), EIs_DCNN)

####################################################################################

