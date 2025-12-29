import os.path
import time
from datetime import datetime

import torch
import numpy as np
import torch.nn as nn
import tqdm
import utils
import matplotlib.pyplot as plt
import guides

class Trainer2D:
    def __init__(self, dtype, device, reference_device, memory_type, sampler, optimizer, scheduler, dx=None, dy=None, num_samples=None, n_lower=None, n_upper=None):
      self.sampler = sampler
      self.optimizer = optimizer
      self.scheduler = scheduler
      self.device = device
      self.reference_device = reference_device
      self.dtype = dtype
      self.n_lower = n_lower
      self.n_upper = n_upper

      if self.n_lower is None:
        self.n_lower = max(self.device.n_cladding, self.device.n_substrate)
        self.n_lower += 0.1 * (self.device.n_core - self.n_lower)
      if self.n_upper is None:
        self.n_upper = self.device.n_core

      self.dx = dx
      self.dy = dy
      self.num_samples = num_samples
      self.memory_type = memory_type

      self._loss_normalizer = 1

    def evaluate(self, x):
        inputs = x
        if self.device.augmented_input and len(x.T)==2:
          features = self.device.make_features(x)
          features.requires_grad = True
          inputs = torch.cat((x, features), dim=1)
        return self.device.underlying_model(inputs)

    @property
    def lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @lr.setter
    def lr(self, value):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = value

    # def relative_loss(self, u,f v):
        # return torch.mean(torch.square(u-v), dim=0, keepdims=True)/(torch.sqrt(torch.mean(torch.square(u), dim=0, keepdims=True))+torch.sqrt(torch.mean(torch.square(v), dim=0, keepdims=True)))
        # return torch.linalg.norm(u-v, dim=0, keepdims=True)/torch.log(1 + torch.linalg.norm(u, dim=0, keepdims=True)+torch.linalg.norm(v, dim=0, keepdims=True))
        # return torch.mean(torch.square(u-v), dim=0, keepdims=True)

    @property
    def loss_normalizer(self):
        return self._loss_normalizer

    @loss_normalizer.setter
    def loss_normalizer(self, value):
        if self._loss_normalizer == 1:
            self._loss_normalizer = value

    ######################################################################################

    def bd_loss(self, u_bd, u_pred_bd, energies):
        return (torch.mean(torch.abs(u_bd - u_pred_bd)))

    ######################################################################################

    def continuity(self, interfaces):
        horizontal_interfaces, vertical_interfaces, horizontal_lengths, vertical_lengths = interfaces

        interface_bias= torch.zeros(sum(horizontal_lengths) + sum(vertical_lengths), 2, dtype=self.dtype, device=self.memory_type)

        interface_bias[:sum(horizontal_lengths), 1] = self.dy
        interface_bias[:horizontal_lengths[0], 1] *= -1

        interface_bias[sum(horizontal_lengths):, 0] = self.dx
        interface_bias[sum(horizontal_lengths):sum(horizontal_lengths)+vertical_lengths[0], 0] *= -1

        interfaces = torch.cat((horizontal_interfaces, vertical_interfaces), dim=0)

        RI_squared_in = torch.square(self.device.refractive_index(interfaces - interface_bias, format='torch')).to(self.memory_type)
        RI_squared_out = torch.square(self.device.refractive_index(interfaces + interface_bias, format='torch')).to(self.memory_type)

        interface_input_in = interfaces - interface_bias
        interface_input_out = interfaces + interface_bias

        output_n_modes_in = self.device.underlying_model(interface_input_in)
        output_n_modes_out = self.device.underlying_model(interface_input_out)

        return interface_input_in, interface_input_out, (output_n_modes_in, output_n_modes_out), (RI_squared_in, RI_squared_out)

    def continuity_augmented(self, interfaces):
        horizontal_interfaces, vertical_interfaces, horizontal_lengths, vertical_lengths = interfaces

        interface_bias= torch.zeros(sum(horizontal_lengths) + sum(vertical_lengths), 2, dtype=self.dtype, device=self.memory_type)

        interface_bias[:sum(horizontal_lengths), 1] = self.dy
        interface_bias[:horizontal_lengths[0], 1] *= -1

        interface_bias[sum(horizontal_lengths):, 0] = self.dx
        interface_bias[sum(horizontal_lengths):sum(horizontal_lengths)+vertical_lengths[0], 0] *= -1

        all_interfaces = torch.cat((horizontal_interfaces, vertical_interfaces), dim=0)

        RI_squared_in = torch.square(self.device.refractive_index(all_interfaces - interface_bias, format='torch')).to(self.memory_type)
        RI_squared_out = torch.square(self.device.refractive_index(all_interfaces + interface_bias, format='torch')).to(self.memory_type)

        features_in =  self.device.make_features(all_interfaces - interface_bias)
        features_out =  self.device.make_features(all_interfaces + interface_bias)

        interface_input_in = torch.cat((all_interfaces, features_in), dim=1)
        interface_input_out = torch.cat((all_interfaces, features_out), dim=1)

        output_n_modes_in = self.device.underlying_model(interface_input_in)
        output_n_modes_out = self.device.underlying_model(interface_input_out)

        return all_interfaces, all_interfaces, (output_n_modes_in, output_n_modes_out), (RI_squared_in, RI_squared_out)

    def continuity_loss(self, interfaces, num_modes, energies):
        horizontal_interfaces, vertical_interfaces, horizontal_lengths, vertical_lengths = interfaces

        random_horizontal = torch.randn(2*len(horizontal_interfaces), 2, dtype=self.dtype, device=self.memory_type)*(self.dx/4)
        random_horizontal[:, 1] = 0
        random_horizontal = random_horizontal[torch.abs(random_horizontal[:, 0]) < self.dx/2][:len(horizontal_interfaces)]
        random_vertical = torch.randn(2*len(vertical_interfaces), 2, dtype=self.dtype, device=self.memory_type)*(self.dy/4)
        random_vertical[:, 0] = 0
        random_vertical = random_vertical[torch.abs(random_vertical[:, 0]) < self.dy/2][:len(vertical_interfaces)]

        if self.device.augmented_input:
            # interface_input_in, interface_input_out, (output_n_modes_in, output_n_modes_out), (RI_squared_in, RI_squared_out) = self.continuity_augmented((horizontal_interfaces, vertical_interfaces, horizontal_lengths, vertical_lengths))
            interface_input_in, interface_input_out, (output_n_modes_in, output_n_modes_out), (RI_squared_in, RI_squared_out) = self.continuity_augmented((horizontal_interfaces + random_horizontal, vertical_interfaces + random_vertical, horizontal_lengths, vertical_lengths))
        else:
            # interface_input_in, interface_input_out, (output_n_modes_in, output_n_modes_out), (RI_squared_in, RI_squared_out) = self.continuity((horizontal_interfaces, vertical_interfaces, horizontal_lengths, vertical_lengths))
            interface_input_in, interface_input_out, (output_n_modes_in, output_n_modes_out), (RI_squared_in, RI_squared_out) = self.continuity((horizontal_interfaces + random_horizontal, vertical_interfaces + random_vertical, horizontal_lengths, vertical_lengths))

        loss_total = torch.zeros(1, device=self.memory_type)

        for i in range(num_modes):
            output_in = output_n_modes_in[:, [i]]
            output_out = output_n_modes_out[:, [i]]

            grad_in = torch.autograd.grad(output_in.sum(), interface_input_in, create_graph=True)[0][:, :2]
            grad_out = torch.autograd.grad(output_out.sum(), interface_input_out, create_graph=True)[0][:, :2]

            loss = torch.zeros(1, device=self.memory_type)
            loss_grad = torch.zeros(1, device=self.memory_type)

            RI_ratio = (RI_squared_in / RI_squared_out)
            serparator_index = sum(horizontal_lengths)

            if self.device.study == 'TM':
                RI_ratio[serparator_index:] = 1
                residuals = (torch.abs(output_out - RI_ratio * output_in))
                grad_residuals = 2*(torch.abs(grad_out - grad_in))
                if not self.device.augmented_input:
                    loss = (torch.mean(residuals[:serparator_index]))
                    loss_grad = (torch.mean(grad_residuals[:serparator_index]))
                else:
                    loss = (torch.mean(residuals))
                    loss_grad = (torch.mean(grad_residuals))

            elif self.device.study == 'TE':
                RI_ratio[:serparator_index] = 1
                residuals = (torch.abs(output_out - RI_ratio * output_in))
                grad_residuals = 2*(torch.abs(grad_out - grad_in))
                if not self.device.augmented_input:
                    loss = (torch.mean(residuals[serparator_index:]))
                    loss_grad = (torch.mean(grad_residuals[serparator_index:]))
                else:
                    loss = (torch.mean(residuals))
                    loss_grad = (torch.mean(grad_residuals))

            elif self.device.study == 'scalar':
                if self.device.augmented_input:
                    loss = (torch.mean(torch.abs(output_out - output_in)))
                    loss_grad = 2*(torch.mean(torch.abs(grad_out - grad_in)))
            else:
                raise Exception('This type of study is not implemented')
            loss_total +=  loss + loss_grad

        return loss_total

    def interior_loss(self, xy, u, energies, num_modes, a, b, eps=1e-5):
        cosine_similarity_fn = nn.CosineSimilarity(dim=0, eps=1e-6)

        losses_1 = torch.zeros(1, device=xy.device)
        losses_2 = torch.zeros(1, device=xy.device)
        losses_3 = torch.zeros(1, device=xy.device)
        losses_4 = torch.zeros(1, device=xy.device)
        rayleigh_values = []

        n_squared = torch.square(self.device.refractive_index(xy, format='torch').detach())[:, 0]

        for i in range(num_modes):
            du = torch.autograd.grad(u[:, i].sum(), xy, create_graph=True, retain_graph=True)[0]

            du_dx = du[:, 0]
            d2u_dx2 = torch.autograd.grad(du_dx.sum(), xy, create_graph=True)[0][:, 0]

            du_dy = du[:, 1]
            d2u_dy2 = torch.autograd.grad(du_dy.sum(), xy, create_graph=True)[0][:, 1]

            rayleigh = guides.RibWaveguideWithPINNs.get_rayleigh_quotient(d2u_dx2, d2u_dy2, n_squared, u[:, i])
            # rayleigh = self.device.underlying_model.eigenvalue
            residual = (torch.mean(torch.abs(d2u_dx2 + d2u_dy2 + (n_squared - rayleigh) * u[:, i])))
            # self.loss_normalizer = residual.item()
            losses_1 += residual
            # losses_1 += torch.mean(torch.abs(du[:, 2]))
            losses_2 += 1/(energies[:, i])
            # losses_2 += torch.square(energies[:, i]-self.n_upper**2)
            # if rayleigh < self.n_lower**2:
            #     losses_3 += (residual + 1e1*(rayleigh<self.n_lower**2))*torch.square(rayleigh - self.n_upper**2)/(self.n_upper**2 - self.n_lower**2)
            # else:
            #     losses_3 += residual/(rayleigh/a - b)
            losses_3 += torch.abs(rayleigh - self.n_lower**2) + torch.abs(rayleigh - self.n_upper**2) - abs(self.n_upper**2 - self.n_lower**2)
            # losses_3 += (residual + 1e1*(rayleigh<self.n_lower**2))*torch.square(rayleigh - self.n_upper**2)/(self.n_upper**2 - self.n_lower**2)
            # losses_3 += (residual + 1e1 * (rayleigh < self.n_lower ** 2)) * torch.abs((rayleigh - self.n_upper ** 2) / (self.n_upper ** 2 - self.n_lower ** 2))
            # losses_3 += (rayleigh + 1e1 * (rayleigh < self.n_upper ** 2)) * torch.abs((rayleigh - self.n_upper ** 2) / (self.n_upper ** 2 - self.n_lower ** 2))
            # losses_3 += torch.abs((rayleigh - self.n_upper ** 2) / (self.n_upper ** 2 - self.n_lower ** 2)) + torch.abs((rayleigh - self.n_lower ** 2) / (self.n_upper ** 2 - self.n_lower ** 2)) - 1
            rayleigh_values.append(rayleigh.detach().item())

            for j in range(i + 1, num_modes):
                losses_4 += torch.abs(cosine_similarity_fn(u[:, i], u[:, j]))

        return losses_1, losses_2, losses_3, losses_4, rayleigh_values

    def construct_training_data(self):
        xy = self.sampler(self.device, self.num_samples if self.num_samples else self.dx,  self.num_samples if self.num_samples else self.dy, self.dtype, self.memory_type)
        if self.device.augmented_input:
            xy = torch.cat((xy, self.device.make_features(xy)), dim=1)
        xy.requires_grad = True
        print('num samples', len(xy))
        return xy

    def train(self, epochs, num_modes, weights=[1, 1, 1, 1, 1, 1], m=2, verbose=0, calc_error=False, plot_solution=False, save_checkpoint=False):
        if verbose ==10:
            loop = tqdm.tqdm(range(epochs))
        else:
            loop = range(epochs)

        # bookkeeping variables
        loss_history = [np.finfo(np.float32).max]
        rayleigh_history = [[0] for _ in range(num_modes)]
        mode_errors_list = []
        for i in range(num_modes):
            mode_errors_list.append([])

        if save_checkpoint:
            now = datetime.now()
            folder_name = now.strftime('%Y-%m-%d %H-%M-%S')
            folder_name = os.path.join('checkpoints', folder_name)
            os.makedirs(folder_name, exist_ok=True)

        # data generation
        self.device.underlying_model.to(self.memory_type)
        self.device.underlying_model.to(self.dtype)

        xy = self.construct_training_data()
        xy.to(self.memory_type)
        interfaces = self.device.get_discontinuities(self.dx, self.dy, self.dtype, self.memory_type)
        total_interface_lengths = 0
        for interface in interfaces:
            if isinstance(interface, torch.Tensor):
                interface.requires_grad=True
                total_interface_lengths+=len(interface)

        bd, u_bd = self.device.get_boundaries(self.dx, self.dy, self.dtype,  self.memory_type)
        plt.figure(figsize=(10, 6))
        plt.scatter(bd[:, 0].cpu(), bd[:, 1].cpu(), s=1)
        plt.scatter(interfaces[0][:, 0].cpu().detach(), interfaces[0][:, 1].cpu().detach(), s=1/4)
        plt.scatter(interfaces[1][:, 0].cpu().detach(), interfaces[1][:, 1].cpu().detach(), s=1/4)
        plt.show()

        # reference solution
        _, xy_fd, mode = self.reference_device.evaluate()
        mode = mode[:, 0]
        levels =  np.linspace(np.min(mode), np.max(mode), num=11)[1:]
        if levels[len(levels)//2]<0:
            mode = -mode
            levels =  np.linspace(np.min(mode), np.max(mode), num=11)[1:]

        # for optimization
        a = (self.n_upper**2 - self.n_lower**2)/(m-1)
        b = (m*self.n_lower**2 - self.n_upper**2)/(self.n_upper**2 - self.n_lower**2)

        # energy_multiplier = torch.cos(torch.linalg.norm(xy[:, :2].detach(), dim=0, keepdims=True)/self.device.total_length_x/np.sqrt(2)*np.pi/2)
        noise_multiplier = torch.tensor([[self.dx/4, self.dy/4, (self.n_upper**2 - self.n_lower**2)/10]], dtype=self.dtype, device=self.memory_type)
        rayleigh_EMA = 1
        corrected_EMA = 1


        for j in loop:
            self.optimizer.zero_grad()
            # if j<epochs-1:
            noise = torch.randn_like(xy)*noise_multiplier
            u_pred = self.evaluate(xy + noise)
            # else:
            #     u_pred = self.evaluate(xy)
            u_pred_bd = self.evaluate(bd)
            energies = torch.sum((u_pred)**2, dim=0, keepdims=True)  * (self.device.total_length_x * self.device.total_length_y) /  len(u_pred)

            loss_5 = self.bd_loss(u_bd, u_pred_bd, energies)
            loss_6 = self.continuity_loss(interfaces, num_modes, energies)


            loss_1, loss_2, loss_3, loss_4, rayleigh_values = self.interior_loss(xy, u_pred, energies, num_modes, a, b)
            loss_history.append((loss_5 + loss_6 + loss_1 + loss_2 + loss_3 + loss_4).detach().item())

            (weights[0]*loss_1 + weights[1]*loss_2+ weights[2]*loss_3+ weights[3]*loss_4+ weights[4]*loss_5 + weights[5]*loss_6).backward()
            # (weights[0]*loss_1 + weights[1]*loss_2+ weights[2]*loss_3+ weights[3]*loss_4+ weights[4]*loss_5 + weights[5]*loss_6*(total_interface_lengths/len(xy))).backward()
            nn.utils.clip_grad_norm_(self.device.underlying_model.parameters(), 1.0)

            for i in range(num_modes):
                rayleigh_history[i].append(rayleigh_values[i])
            if verbose == 10:
                loop.set_description(f'loss1 is {round(loss_1.detach().item(), 5):8.5f}, loss2 is {round(loss_2.detach().item(), 5):8.5f}, loss3 is {round(loss_3.detach().item(), 5):8.5f}, loss4 is {round(loss_4.detach().item(), 5):8.5f}, loss5 is {round(loss_5.detach().item(), 5):8.5f}, loss6 is {round(loss_6.detach().item(), 5):8.5f}, total loss {round(loss_history[-1], 5):8.5f}, RI value {round(np.sqrt(rayleigh_history[0][-1]), 5):8.5f}')
            elif verbose > 0:
                if j%(int(1000/verbose))==0:
                  print(f'iteration {j+1} loss1 is {round(loss_1.detach().item(), 5):8.5f}, loss2 is {round(loss_2.detach().item(), 5):8.5f}, loss3 is {round(loss_3.detach().item(), 5):8.5f}, loss4 is {round(loss_4.detach().item(), 5):8.5f}, loss5 is {round(loss_5.detach().item(), 5):8.5f}, loss6 is {round(loss_6.detach().item(), 5):8.5f}, total loss {round(loss_history[-1], 5):8.5f}, RI value {round(np.sqrt(rayleigh_history[0][-1]), 5):8.5f}')

                # return (weights[0]*loss_1 + weights[1]*loss_2+ weights[2]*loss_3+ weights[3]*loss_4+ weights[4]*loss_5 +weights[5]*loss_6)

            self.optimizer.step()
            self.scheduler.step()

            # new_EMA =  (0.99 * rayleigh_EMA + 0.01 * rayleigh_values[-1])
            # new_corrected_EMA = new_EMA / (1-(0.99)**(j+1))
            # if abs(new_corrected_EMA - corrected_EMA) < 1e-5:
            # if abs(new_EMA - rayleigh_EMA) < 1e-6:
            #     print('Converged')
            #     if verbose == 10:
            #         loop.close()
            #     return loss_history, rayleigh_history, mode_errors_list
            # rayleigh_EMA = new_EMA
            # corrected_EMA = new_corrected_EMA

            if verbose > 0:
                if j%(int(5000/verbose))==0:

                    with torch.no_grad():
                        xy_slice = torch.arange(0, self.device.total_length_x / 2, self.dx, dtype=self.dtype, device=self.memory_type).reshape(-1, 1)
                        xy_slice = torch.cat((-torch.flip(xy_slice, dims=(0,))[:-1], xy_slice))
                        xy_slice = torch.cat((xy_slice, torch.full_like(xy_slice, 0 * self.dx)), dim=-1)

                        u_slice = self.evaluate(xy_slice).cpu()

                        u = self.evaluate(xy)
                        u = u / torch.linalg.norm(u, dim=0, keepdims=True)

                    if plot_solution:
                        u_levels =torch.linspace(u.min(), u.max(), 11)
                        mode_levels = torch.linspace(mode.min(), mode.max(), steps=11)
                        plt.tricontour(xy[:, 0].detach().cpu(), xy[:, 1].detach().cpu(), u[:, 0].cpu().view(-1), cmap='jet',levels=u_levels)
                        plt.tricontour(xy_fd[:, 0], xy_fd[:, 1], mode, colors='black', linestyles='dashed', levels=mode_levels)
                        core_length_x = self.device.lengths_x[1]
                        core_length_y = self.device.lengths_y[1]
                        plt.vlines(-core_length_x / 2, -core_length_y / 2, core_length_y / 2, colors='black')
                        plt.vlines(core_length_x / 2, -core_length_y / 2, core_length_y / 2, colors='black')
                        plt.hlines(-core_length_y / 2, -core_length_x / 2, core_length_x / 2, colors='black')
                        plt.hlines(core_length_y / 2, -core_length_x / 2, core_length_x / 2, colors='black')

                        plt.grid()
                        plt.show()
                        plt.scatter(xy_slice[:, 0].cpu().detach(), u_slice[:, 0].detach().numpy(), s=1)
                        plt.vlines(x = self.device.lengths_x[1]/2, ymin = -1, ymax=1, color='r', linewidth=1)
                        plt.vlines(x = -self.device.lengths_x[1]/2, ymin = -1, ymax=1, color='r', linewidth=1)
                        plt.grid()
                        plt.show()

                    if calc_error:
                        mode_errors = utils.compare_against_analytic(self.device, xy[:, [0]].clone().detach().cpu().numpy(), u.cpu())
                        for i in range(num_modes):
                            mode_errors_list[i].append(mode_errors[i])

            if save_checkpoint and (j+1)%1000==0:
                torch.save(self.device.underlying_model.state_dict(), os.path.join(os.getcwd(), folder_name, f'iteration{j+1}Model.pth'))
                torch.save(self.optimizer.state_dict(), os.path.join(os.getcwd(), folder_name, f'iteration{j+1}Optimizer.pth'))
                torch.save(self.scheduler.state_dict(), os.path.join(os.getcwd(), folder_name, f'iteration{j+1}Scheduler.pth'))

                with open(os.path.join(os.getcwd(), folder_name, f'Checkpoint.txt'), 'a') as f:
                    f.write(
                        f'iteration {j + 1} loss1 is {round(loss_1.detach().item(), 5):8.5f}, loss2 is {round(loss_2.detach().item(), 5):8.5f}, loss3 is {round(loss_3.detach().item(), 5):8.5f}, loss4 is {round(loss_4.detach().item(), 5):8.5f}, loss5 is {round(loss_5.detach().item(), 5):8.5f}, loss6 is {round(loss_6.detach().item(), 5):8.5f}, total loss {round(loss_history[-1], 5):8.5f}, RI value {round(np.sqrt(rayleigh_history[0][-1]), 5):8.5f}\n')


        if verbose == 10:
          loop.close()

        return loss_history, rayleigh_history, mode_errors_list

