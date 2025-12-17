import os, sys
os.chdir(r"C:\Users\asmaa\Desktop\Amr\Master's Thesis v0.4")
sys.path.append(r"C:\Users\asmaa\Desktop\Amr\Master's Thesis v0.4")

import devices
import models
import samplers
import trainers
import utils

import torch
import numpy as np

# parameters
lambd = 1.55

# wave number and normalized spacing
k_0 = 2 * np.pi / lambd


# model hyperparameters
weights = [1/2, 1/4, 8, 1, 1, 1]
m = 2

# simulation device parameters
memory_type = torch.device('cuda')
dtype = torch.float64

# number of modes
num_modes = 1
###############################################################################
# # device 1
# # simulation lengths
core_length_x = 2 * k_0 # W
core_length_y = 1 * k_0 # H

substrate_length_x = 3.0 * k_0 # Xs
substrate_length_y = 4.0 * k_0 # Ys

# refractive indices
n_core = 1.55
n_substrate = 1.44

# Preparing simulation parameters
lengths_x = [substrate_length_x, core_length_x, substrate_length_x]
lengths_y = [substrate_length_y, core_length_y, substrate_length_y]
RIs = [n_substrate, n_core]

# # ###############################################################################
#
# # study and field types
# study = 'TE'
# field_type = 'E'
#
#
# reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')
#
# save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TE"
#
# EIs_FD = []
# modes_errors_FD = []
#
#
# for i in range(24, 25):
#     dx = k_0 / ((i+1)*5) # For device 1
#     dy = k_0 / ((i+1)*10) # For device 1
#
#     # FD
#     fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
#     RI_squared, x, u = fd_device.evaluate()
#     u = u / np.max(np.abs(u), axis=0, keepdims=True)
#     EIs_FD.append(RI_squared.item()**(1/2))
#
#     print(EIs_FD)
#
#
# np.save(save_folder_path + r"\NonPlanar\Single Mode Weakly Guiding\Convergence\EIs_FD.npy", EIs_FD)

###############################################################################

# study and field types
study = 'TM'

# Preparing simulation parameters
reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')

save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TM"

EIs_FD = []
modes_errors_FD = []


for i in range(20):
    dx = k_0 / ((i+1)*5) # For device 1
    dy = k_0 / ((i+1)*10) # For device 1

    # FD
    fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
    RI_squared, x, u = fd_device.evaluate()
    u = u / np.max(np.abs(u), axis=0, keepdims=True)
    import matplotlib.pyplot as plt
    plt.scatter(x[:, 0], x[:, 1], c=u, cmap='inferno')
    plt.show()
    EIs_FD.append(RI_squared.item()**(1/2))

    print(EIs_FD)


np.save(save_folder_path + r"\NonPlanar\Single Mode Weakly Guiding\Convergence\EIs_FD.npy", EIs_FD)

# ###############################################################################
#
# # device 2
# # simulation lengths
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
#
# ###############################################################################
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
# for i in range(20):
#     dx = k_0 / ((2*i+1)*5) # For device 2
#     dy = k_0 / ((2*i+1)*5) # For device 2
#
#     # FD
#     fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
#     RI_squared, x, u = fd_device.evaluate()
#     u = u / np.max(np.abs(u), axis=0, keepdims=True)
#     EIs_FD.append(RI_squared.item()**(1/2))
#
#     print(EIs_FD)
#
#
# np.save(save_folder_path + r"\NonPlanar\Single Mode Moderate Contrast\Convergence\EIs_FD.npy", EIs_FD)
# save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TM"
# np.save(save_folder_path + r"\NonPlanar\Single Mode Moderate Contrast\Convergence\EIs_FD.npy", EIs_FD)

# ###############################################################################
#
# # # study and field types
# # study = 'TM'
# #
# # # Preparing simulation parameters
# # reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')
# #
# # save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TM"
# #
# # EIs_FD = []
# # modes_errors_FD = []
# #
# # EIs_DCNN = []
# # modes_errors_DCNN = []
# #
# # for i in range(10):
# #     # dx = k_0 / ((2*i+1)*10) # For device 3
# #     # dy = k_0 / ((2*i+1)*10) # For device 3
# #     dx = k_0 / ((i+1)*20) # For device 2
# #     dy = k_0 / ((i+1)*20) # For device 2
# #
# #     # FD
# #     fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
# #     RI_squared, x, u = fd_device.evaluate()
# #     u = u / np.max(np.abs(u), axis=0, keepdims=True)
# #     EIs_FD.append(RI_squared.item()**(1/2))
# #
# #     print(EIs_FD)
#
# # np.save(save_folder_path + r"\NonPlanar\Single Mode Moderate Contrast\Convergence\EIs_FD.npy", EIs_FD)
# #
# #####################################################################################

# device 3
# simulation lengths
# core_length_x = 1 * k_0 # W
# core_length_y = 2 * k_0 # H
#
# substrate_length_x = 2 * k_0 # Xs
# substrate_length_y = 1 * k_0 # Ys
#
# # refractive indices
# n_core = 3
# n_substrate = 1
#
# # study and field types
# # study = 'TE'
# field_type = 'E'
# #
# # Preparing simulation parameters
# lengths_x = [substrate_length_x, core_length_x, substrate_length_x]
# lengths_y = [substrate_length_y, core_length_y, substrate_length_y]
# RIs = [n_substrate, n_core]
# # reference_device = devices.BuriedChannelWaveguide(RIs, lambd, lengths_x, lengths_y, study, 'E')
#
# # save_folder_path = r"C:\Users\asmaa\Desktop\Amr\2nd Paper\Experiments\TE"
# #
# # EIs_FD = []
# # modes_errors_FD = []
# #
# # for i in range(15):
# #     dx = k_0 / ((2*i+1)*5) # For device 3
# #     dy = k_0 / ((2*i+1)*5) # For device 3
# #
# #     # FD
# #     fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
# #     RI_squared, x, u = fd_device.evaluate()
# #     u = u / np.max(np.abs(u), axis=0, keepdims=True)
# #     EIs_FD.append(RI_squared.item()**(1/2))
# #
# #     print(EIs_FD)
# #
# #
# # np.save(save_folder_path + r"\NonPlanar\Single Mode High Contrast\Convergence\EIs_FD.npy", EIs_FD)
# # #
# ###############################################################################
#
# # # study and field types
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
# for i in range(19, 20):
#     dx = k_0 / ((2*i+1)*5) # For device 3
#     dy = k_0 / ((2*i+1)*5) # For device 3
#
#     # FD
#     fd_device = devices.BuriedChannelWaveguideWithFD(RIs, lambd, lengths_x, lengths_y, study, 'E', dx, dy, num_modes=num_modes)
#     RI_squared, x, u = fd_device.evaluate()
#     u = u / np.max(np.abs(u), axis=0, keepdims=True)
#     EIs_FD.append(RI_squared.item()**(1/2))
#
#     print(EIs_FD)
#
# np.save(save_folder_path + r"\NonPlanar\Single Mode High Contrast\Convergence\EIs_FD.npy", EIs_FD)
#
#####################################################################################

