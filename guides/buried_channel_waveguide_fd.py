from math import sqrt
import torch
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from .buried_channel_waveguide import BuriedChannelWaveguide

class BuriedChannelWaveguideWithFD(BuriedChannelWaveguide):
  
    def __init__(self, RIs, lambd, lengths_x, lengths_y, study, field_type, dx, dy, num_modes=1):
        BuriedChannelWaveguide.__init__(self, RIs, lambd, lengths_x, lengths_y, study, field_type)
        self.dx = dx
        self.dy = dy
        self.num_modes = num_modes

    def assemble_derivative_matrix_x(self, num_nodes, Ny, bd_pts=None):
        diagonal_elements = np.ones((num_nodes, )) * -2
        off_diagonal_elements_u = np.ones((num_nodes - 1, ))
        off_diagonal_elements_l = np.ones((num_nodes - 1, ))
        if bd_pts is not None:
            off_diagonal_elements_u[np.sort(bd_pts)[:-1]]=0
            off_diagonal_elements_l[(np.sort(bd_pts)-1)[1:]]=0
        d2u_dx2 = diags([diagonal_elements, off_diagonal_elements_l, off_diagonal_elements_u], [0, -1, 1], format='csc')
        d2u_dx2 /= self.dx ** 2
        return d2u_dx2

    def assemble_derivative_matrix_y(self, num_nodes, Nx, bd_pts=None):
        diagonal_elements = np.ones((num_nodes, )) * -2
        off_diagonal_elements_u = np.ones((num_nodes - Nx, ))
        off_diagonal_elements_l = np.ones((num_nodes - Nx, ))
        if bd_pts is not None:
            off_diagonal_elements_u[np.sort(bd_pts)[:-Nx]]=0
            off_diagonal_elements_l[(np.sort(bd_pts)-Nx)[Nx:]]=0
        d2u_dy2 = diags([diagonal_elements, off_diagonal_elements_l, off_diagonal_elements_u], [0, -Nx, Nx], format='csc')
        d2u_dy2 /= self.dy ** 2
        return d2u_dy2

    def assemble_scale_matrix(self, xy, bd_pts=None):
        diagonal_elements = self.refractive_index(xy, format='numpy')[:, 0]
        if bd_pts is not None:
            diagonal_elements[bd_pts]=0
        n_x = diags([diagonal_elements], [0], format='csc')
        return n_x

    def make_grid(self):
        xr = np.arange(0, self.simulation_lengths_x[0], self.dx).reshape(-1, 1)
        xl = np.arange(0, self.simulation_lengths_x[1], self.dx).reshape(-1, 1)
        x = np.concatenate((-xl[::-1][:-1], xr))
        yu = np.arange(0, self.simulation_lengths_y[0], self.dy).reshape(-1, 1)
        yd = np.arange(0, self.simulation_lengths_y[1], self.dy).reshape(-1, 1)
        y = np.concatenate((-yd[::-1][:-1], yu))
        Nx = len(x)
        Ny = len(y)
        x = np.tile(x, Ny).T.reshape(-1, 1)
        y = np.tile(y, Nx).reshape(-1, 1)
        xy = np.concatenate((x, y), axis=1)
        return xy, Nx, Ny

    def assemble_problem_matrix(self, xy, num_nodes, Nx, Ny, bd_pts=None):
        mask_x = np.zeros_like(xy)
        mask_x[:, 0] = 1
        mask_y = np.zeros_like(xy)
        mask_y[:, 1] = 1

        pq = np.square(self.refractive_index(xy, format='numpy'))[:, 0]
        pq_prev_p = np.square(self.refractive_index(xy-self.dx*mask_x, format='numpy'))[:, 0]
        pq_post_p = np.square(self.refractive_index(xy+self.dx*mask_x, format='numpy'))[:, 0]
        pq_prev_q = np.square(self.refractive_index(xy-self.dy*mask_y, format='numpy'))[:, 0]
        pq_post_q = np.square(self.refractive_index(xy+self.dy*mask_y, format='numpy'))[:, 0]

        e = w = self.dx
        s = n = self.dy
        if self.study=='TE':
            alpha_w = ((2/(w*(e+w))) * (2*pq_prev_p)/(pq + pq_prev_p)) # xl
            alpha_e = ((2/(e*(e+w))) * (2*pq_post_p)/(pq + pq_post_p)) # xr
            alpha_n = 2/(n*(n+s)) * np.ones(num_nodes - Nx) # yl
            alpha_s = 2/(s*(n+s)) * np.ones(num_nodes - Nx) # yu
            alpha_x = -(4/(e*w)) + alpha_e + alpha_w # xy
            alpha_y = -(2/(n*s)) * np.ones_like(alpha_x) # xy
        else:
            alpha_w = (2/(w*(e+w)))  * np.ones(num_nodes - 1) # xl
            alpha_e = (2/(e*(e+w))) * np.ones(num_nodes - 1) # xr
            alpha_n = 2/(n*(n+s)) * (2*pq_prev_q)/(pq + pq_prev_q)  # yl
            alpha_s = 2/(s*(n+s)) * (2*pq_post_q)/(pq + pq_post_q) # yu
            alpha_y = -(4/(n*s)) + alpha_n + alpha_s # xy
            alpha_x = -(2/(e*w)) * np.ones_like(alpha_y)  # xy

        if bd_pts is not None:
            alpha_e[np.sort(bd_pts)[:-1]]=0
            alpha_w[(np.sort(bd_pts)-1)[1:]]=0

            alpha_s[np.sort(bd_pts)[:-Nx]]=0
            alpha_n[(np.sort(bd_pts)-Nx)[Nx:]]=0

            alpha_x[bd_pts]=1
            alpha_y[bd_pts]=0
            pq[bd_pts]=0

        if self.study=='TE':
            A = diags([(alpha_x + alpha_y + pq), alpha_w[1:], alpha_e[:-1], alpha_n, alpha_s], [0, -1, 1, -Nx, Nx])
        else:
            A = diags([(alpha_x + alpha_y + pq), alpha_w, alpha_e, alpha_n[Nx:], alpha_s[:-Nx]], [0, -1, 1, -Nx, Nx])
            
        return A

    def evaluate(self):
        xy, Nx, Ny = self.make_grid()
        num_nodes = len(xy)

        bd_pts_xlower = np.arange(Nx)
        bd_pts_xupper = Nx*(Ny-1) + np.arange(Nx)
        bd_pts_yleft = np.arange(Ny) * Nx
        bd_pts_yright = np.arange(Ny) * Nx + Nx - 1
        bd_pts = np.concatenate((bd_pts_xlower, bd_pts_xupper, bd_pts_yleft[1:-1], bd_pts_yright[1:-1]))

        if self.study =='scalar':
            d2u_dx2 = self.assemble_derivative_matrix_x(num_nodes, Ny, bd_pts)
            d2u_dy2 = self.assemble_derivative_matrix_y(num_nodes, Nx, bd_pts)
            n_u = self.assemble_scale_matrix(xy, bd_pts)
            A = d2u_dx2 + d2u_dy2 +  n_u ** 2
        else:
            A = self.assemble_problem_matrix(xy, num_nodes, Nx, Ny, bd_pts)
        eig_vals, eig_vecs = eigs(A, k=self.num_modes, sigma=self.n_core**2)
        eig_vals = eig_vals.real
        eig_vecs = eig_vecs.real
        sorting_indices = np.argsort(-eig_vals)
        return eig_vals[sorting_indices[:self.num_modes]], xy, eig_vecs[:, sorting_indices[:self.num_modes]]