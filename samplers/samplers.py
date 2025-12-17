# from math import sqrt
# import torch

# def uniform_grid_2d(device, dx, dy, dtype, memory_type):
#     xu = torch.arange(0, device.simulation_lengths_x[0], dx, dtype=dtype, device=memory_type).reshape(-1, 1)
#     xl = torch.arange(0, device.simulation_lengths_x[1], dx, dtype=dtype, device=memory_type).reshape(-1, 1)
#     x = torch.cat((torch.flip(-xl, dims=(0,))[:-1], xu))
#     # x = torch.linspace(-device.simulation_lengths_x[0], device.simulation_lengths_x[1], int(sum(device.simulation_lengths_x)//dx)+1, dtype=dtype, device=memory_type)

#     yu = torch.arange(0, device.simulation_lengths_y[0], dy, dtype=dtype, device=memory_type).reshape(-1, 1)
#     yl = torch.arange(0, device.simulation_lengths_y[1], dy, dtype=dtype, device=memory_type).reshape(-1, 1)
#     y = torch.cat((torch.flip(-yl, dims=(0,))[:-1], yu))

#     Nx = len(x)
#     Ny = len(y)

#     x = x.repeat(1, Ny).T.reshape(-1, 1)
#     y = y.repeat(1, Nx).reshape(-1, 1)
#     xy = torch.cat((x, y), axis=1)
    
#     return xy

# def uniform_grid_2d_customized(device, dx, dy, dtype, memory_type, start_value_x=0, start_value_y=0, end_value_x=None, end_value_y=None):
#     x = torch.arange(start_value_x, end_value_x, dx, dtype=dtype, device=memory_type).reshape(-1, 1)
#     end_value_x = x[-1].item()
#     x = torch.cat((torch.flip(-x, dims=(0,))[:-1], x))

#     y = torch.arange(start_value_y, end_value_y, dy, dtype=dtype, device=memory_type).reshape(-1, 1)
#     end_value_y = y[-1].item()
#     y = torch.cat((torch.flip(-y, dims=(0,))[:-1], y))

#     Nx = len(x)
#     Ny = len(y)

#     x = x.repeat(1, Ny).T.reshape(-1, 1)
#     y = y.repeat(1, Nx).reshape(-1, 1)
#     xy = torch.cat((x, y), axis=1)
    
#     return xy, end_value_x, end_value_y

# def stepped_grid_2d(device, dx, dy, dtype, memory_type):
#     dx *= 1.25
#     dy *= 1.25

#     xy_dense, end_value_x, end_value_y = uniform_grid_2d_customized(device, dx/2, dy/2, dtype, memory_type, start_value_x=0, start_value_y=0, end_value_x=device.central_region[0]+10*dx, end_value_y=device.central_region[1]+10*dy)
#     xy_sparse, _, __ = uniform_grid_2d_customized(device, dx, dy, dtype, memory_type, start_value_x=0, start_value_y=0, end_value_x=device.total_length_x/2, end_value_y=device.total_length_y/2)
#     mask_sparse = (xy_sparse[:, 0] > device.central_region[0] + 10*dx) & (xy_sparse[:, 1] > device.central_region[1] + 10*dy) 

#     return torch.cat((xy_dense, xy_sparse))

# def uniform_grid(device, dx, dtype, memory_type):
#     X = torch.arange(0, device.total_length/2, dx, dtype=dtype, device=memory_type).reshape(-1, 1)
#     X = torch.cat((torch.flip(-X, dims=(0,))[:-1], X))
#     return X

# def stepped_grid(device, dx, dtype, memory_type):
#     dx *= 1.18
#     x = torch.arange(0, device.central_region + 10*dx, dx/2, dtype=dtype, device=memory_type).reshape(-1, 1)
#     x = torch.cat((torch.flip(-x, dims=(0,))[:-1], x))
#     X = torch.arange(x[-1].item() + dx, device.total_length/2, dx, dtype=dtype, device=memory_type).reshape(-1, 1)
#     X = torch.cat((torch.flip(-X, dims=(0,))[:-1], X))
#     return torch.cat((x, X))

# def normally_distributed_grid(device, num_samples, dtype, memory_type, std=None):
#     if std==None:
#       std = sqrt(device.central_region * 10)
#     initial =  torch.randn(3 * num_samples, 1, dtype=dtype, device=memory_type)*std
#     initial = torch.sort(initial[((initial>0) & (initial<device.total_length/2))[:, 0]][:int(num_samples//2)], dim=0)[0]
#     return torch.cat((torch.flip(-initial, dims=(0,)), initial))

# def uniformly_distibuted_grid(device, num_samples, dtype, memory_type):
#       return torch.quasirandom.SobolEngine(dimension=1, scramble=True).draw(num_samples, dtype=dtype).to(memory_type) * device.total_length - device.total_length/2
from math import sqrt
import torch

def uniform_grid_2d(device, dx, dy, dtype, memory_type):
    xu = torch.arange(0, device.simulation_lengths_x[0], dx, dtype=dtype, device=memory_type).reshape(-1, 1)
    xl = torch.arange(0, device.simulation_lengths_x[1], dx, dtype=dtype, device=memory_type).reshape(-1, 1)
    x = torch.cat((torch.flip(-xl, dims=(0,))[:-1], xu))

    yu = torch.arange(0, device.simulation_lengths_y[0], dy, dtype=dtype, device=memory_type).reshape(-1, 1)
    yl = torch.arange(0, device.simulation_lengths_y[1], dy, dtype=dtype, device=memory_type).reshape(-1, 1)
    y = torch.cat((torch.flip(-yl, dims=(0,))[:-1], yu))

    Nx = len(x)
    Ny = len(y)

    x = x.repeat(1, Ny).T.reshape(-1, 1)
    y = y.repeat(1, Nx).reshape(-1, 1)
    xy = torch.cat((x, y), axis=1)
    
    return xy

def uniform_grid_2d_customized(device, dx, dy, dtype, memory_type, start_value_x=0, start_value_y=0, end_value_x=None, end_value_y=None):
    x = torch.arange(start_value_x, end_value_x, dx, dtype=dtype, device=memory_type).reshape(-1, 1)
    end_value_x = x[-1].item()
    x = torch.cat((torch.flip(-x, dims=(0,))[:-1], x))

    y = torch.arange(start_value_y, end_value_y, dy, dtype=dtype, device=memory_type).reshape(-1, 1)
    end_value_y = y[-1].item()
    y = torch.cat((torch.flip(-y, dims=(0,))[:-1], y))

    Nx = len(x)
    Ny = len(y)

    x = x.repeat(1, Ny).T.reshape(-1, 1)
    y = y.repeat(1, Nx).reshape(-1, 1)
    xy = torch.cat((x, y), axis=1)
    
    return xy, end_value_x, end_value_y

def stepped_grid_2d(device, dx, dy, dtype, memory_type):
    dx *= 1.25
    dy *= 1.25

    xy_dense, end_value_x, end_value_y = uniform_grid_2d_customized(device, dx/2, dy/2, dtype, memory_type, start_value_x=0, start_value_y=0, end_value_x=device.central_region[0]+10*dx, end_value_y=device.central_region[1]+10*dy)
    xy_sparse, _, __ = uniform_grid_2d_customized(device, dx, dy, dtype, memory_type, start_value_x=0, start_value_y=0, end_value_x=device.total_length_x/2, end_value_y=device.total_length_y/2)
    mask_sparse = (xy_sparse[:, 0] > device.central_region[0] + 10*dx) & (xy_sparse[:, 1] > device.central_region[1] + 10*dy) 

    return torch.cat((xy_dense, xy_sparse))

def uniform_grid(device, dx, dtype, memory_type):
    X = torch.arange(0, device.total_length/2, dx, dtype=dtype, device=memory_type).reshape(-1, 1)
    X = torch.cat((torch.flip(-X, dims=(0,))[:-1], X))
    return X

def stepped_grid(device, dx, dtype, memory_type):
    dx *= 1.18
    x = torch.arange(0, device.central_region + 10*dx, dx/2, dtype=dtype, device=memory_type).reshape(-1, 1)
    x = torch.cat((torch.flip(-x, dims=(0,))[:-1], x))
    X = torch.arange(x[-1].item() + dx, device.total_length/2, dx, dtype=dtype, device=memory_type).reshape(-1, 1)
    X = torch.cat((torch.flip(-X, dims=(0,))[:-1], X))
    return torch.cat((x, X))

def normally_distributed_grid(device, num_samples, dtype, memory_type, std=None):
    if std==None:
      std = sqrt(device.central_region * 10)
    initial =  torch.randn(3 * num_samples, 1, dtype=dtype, device=memory_type)*std
    initial = torch.sort(initial[((initial>0) & (initial<device.total_length/2))[:, 0]][:int(num_samples//2)], dim=0)[0]
    return torch.cat((torch.flip(-initial, dims=(0,)), initial))

def uniformly_distibuted_grid(device, num_samples, dtype, memory_type):
      return torch.quasirandom.SobolEngine(dimension=1, scramble=True).draw(num_samples, dtype=dtype).to(memory_type) * device.total_length - device.total_length/2
