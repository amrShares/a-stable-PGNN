from torch.optim import Optimizer
import torch

class LevenbergMarquardt(Optimizer):
    r"""Implements the Levenberg-Marquardt algorithm.

    It has been proposed in "A method for non-linear least squares problems"
    by Levenberg (1944) and independently by Marquardt (1963).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        lambd (float, optional): initial value of the damping parameter lambda
            (default: 1e-3)
        lambd_up_factor (float, optional): factor to increase lambda when loss
            increases (default: 2.0)
        lambd_down_factor (float, optional): factor to decrease lambda when loss
            decreases (default: 3.0)
        min_lambd (float, optional): minimum value of lambda (default: 1e-6)
        max_lambd (float, optional): maximum value of lambda (default: 1e6)
        eps (float, optional): term added to the diagonal of the approximate
            Hessian to improve numerical stability (default: 1e-8)

    """

    def __init__(self, params, lr=1e-3, lambd=1e-3, lambd_up_factor=2.0,
                 lambd_down_factor=3.0, min_lambd=1e-6, max_lambd=1e6, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lambd:
            raise ValueError("Invalid lambda value: {}".format(lambd))
        if not 1.0 <= lambd_up_factor:
            raise ValueError("Invalid lambda up factor: {}".format(lambd_up_factor))
        if not 1.0 <= lambd_down_factor:
            raise ValueError("Invalid lambda down factor: {}".format(lambd_down_factor))
        if not 0.0 <= min_lambd:
            raise ValueError("Invalid minimum lambda: {}".format(min_lambd))
        if not 0.0 <= max_lambd:
            raise ValueError("Invalid maximum lambda: {}".format(max_lambd))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, lambd=lambd, lambd_up_factor=lambd_up_factor,
                        lambd_down_factor=lambd_down_factor, min_lambd=min_lambd,
                        max_lambd=max_lambd, eps=eps)
        super(LevenbergMarquardt, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['prev_loss'] = None

    @torch.no_grad()
    def step(self, closure=None):
        r"""Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                state['step'] += 1
                lr = group['lr']
                lambd = group['lambd']
                lambd_up_factor = group['lambd_up_factor']
                lambd_down_factor = group['lambd_down_factor']
                min_lambd = group['min_lambd']
                max_lambd = group['max_lambd']
                eps = group['eps']

                grad = p.grad.data.flatten()
                numel = p.numel()

                # Approximate Hessian (Gauss-Newton approximation: J^T J)
                # For efficiency, we use the squared gradient as a diagonal approximation
                approx_hessian = grad.pow(2) + eps * torch.eye(numel, device=grad.device)

                # Levenberg-Marquardt update rule
                delta = torch.linalg.solve(approx_hessian + lambd * torch.eye(numel, device=grad.device), -lr * grad.unsqueeze(1)).squeeze(1)

                # Reshape the update to the parameter's original shape
                delta = delta.reshape_as(p.data)

                # Perform the update (tentatively)
                p.data.add_(delta)

                if closure is not None:
                    with torch.enable_grad():
                        new_loss = closure()

                    if state['prev_loss'] is None or new_loss < state['prev_loss']:
                        # Accept the update
                        state['prev_loss'] = new_loss
                        group['lambd'] = max(min_lambd, lambd / lambd_down_factor)
                    else:
                        # Reject the update, revert parameters and increase lambda
                        p.data.sub_(delta)
                        group['lambd'] = min(max_lambd, lambd * lambd_up_factor)
                else:
                    state['prev_loss'] = loss  # Update previous loss even without closure

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This is more memory efficient, and can speed up subsequently calls
                to :meth:`torch.optim.Optimizer.step`.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()