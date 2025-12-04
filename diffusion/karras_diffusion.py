"""
Based on: https://gitorchub.com/crowsonkb/k-diffusion
"""

import math
import numpy as np
import torch
from piq import LPIPS
from tqdm.auto import tqdm
import torch.distributed as dist
from copy import deepcopy
import gc
from scipy.special import spence

from .nn import mean_flat, append_dims, append_zero
from .random_util import BatchedSeedGenerator

class NoiseSchedule:
    def __init__(self):
        raise NotImplementedError

    def get_f_g2(self, t):
        raise NotImplementedError

    def get_alpha_rho(self, t):
        raise NotImplementedError

    def get_abc(self, t):
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        
        a_t, b_t, c_t = (
            (alpha_bar_t * rho_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t * rho_t) / self.rho_T,
        )
        return a_t, b_t, c_t

    def get_alpha_prime(self, t):
        raise NotImplementedError
    
    def get_sigma_prime(self, t):
        raise NotImplementedError
    
    def get_abc_prime(self, t, a_t=None, b_t=None, c_t=None):
        alpha_T, rho_T = self.alpha_T, self.rho_T
        sigma_T = alpha_T * rho_T
        snr_T = 1 / rho_T ** 2
        
        alpha_t = self.alpha_fn(t)
        alpha_t_prime = self.get_alpha_prime(t)
        rho_t = self.rho_fn(t)
        sigma_t = alpha_t * rho_t
        sigma_t_prime = self.get_sigma_prime(t)
        
        if (a_t is None) or (b_t is None) or (c_t is None):
            a_t, b_t, c_t = self.get_abc(t)
            # a_t, b_t, c_t = self.get_abc(t.to(torch.float64))
        
        common = (2 * sigma_t_prime / sigma_t) - (alpha_t_prime / alpha_t)

        a_t_prime = a_t * common
        
        b_t_prime = common * snr_T * sigma_t.square() / alpha_t
        b_t_prime = alpha_t_prime - b_t_prime
        
        c_t_mult_c_t_prime = common * snr_T * (sigma_t ** 4) / (alpha_t ** 2)
        c_t_mult_c_t_prime = (sigma_t * sigma_t_prime) - c_t_mult_c_t_prime
        
        return a_t_prime, b_t_prime, c_t_mult_c_t_prime
    
    def get_timestep_from_lambda(self, lambda_t):
        raise NotImplementedError

class VPNoiseSchedule(NoiseSchedule):
    def __init__(self, beta_d=2, beta_min=0.1):
        self.name = "VP"
        self.sigma_max = 1
        self.beta_d, self.beta_min = beta_d, beta_min
        self.alpha_fn = lambda t: np.e ** (-0.5 * beta_min * t - 0.25 * beta_d * t**2)
        self.alpha_T = self.alpha_fn(1)
        self.rho_fn = lambda t: (np.e ** (beta_min * t + 0.5 * beta_d * t**2) - 1).sqrt()
        self.rho_T = self.rho_fn(torch.DoubleTensor([self.sigma_max])).item()

        self.lambda_inverse = lambda l: (-beta_min + ((beta_min**2) + (2 * beta_d * torch.log(1 + (-2*l).exp()))).sqrt()) / beta_d
        
        self.f_fn = lambda t: (-0.5 * beta_min - 0.5 * beta_d * t)
        self.g2_fn = lambda t: (beta_min + beta_d * t)
        
    def get_alpha_prime(self, t):
        t = t.to(torch.float64)
        return - 0.5 * self.alpha_fn(t) * (self.beta_min + (self.beta_d * t))

    def get_sigma_prime(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_t_sq = alpha_t.square()
        alpha_t_prime = self.get_alpha_prime(t)
        return alpha_t * alpha_t_prime / (1. - alpha_t_sq).sqrt()
        
    def get_timestep_from_lambda(self, lambda_t):
        return self.lambda_inverse(lambda_t.to(torch.float64)).to(torch.float32)
    
    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        # print('t:', t.unique())
        # print('alpha_t:', alpha_t.unique())
        # print('alpha_T:', self.alpha_T)
        alpha_bar_t = alpha_t / self.alpha_T
        # print('alpha_bar_t:', alpha_bar_t.unique().item())
        rho_t = self.rho_fn(t)
        # print('rho_t:', rho_t.unique().item())
        # print('rho_T', self.rho_T)
        rho_bar_t = torch.sqrt(self.rho_T**2 - rho_t**2)
        # print('rho_bar_t:', (self.rho_T**2 - rho_t**2).unique())
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class VENoiseSchedule(NoiseSchedule):
    def __init__(self, sigma_max=80.0):
        self.name = "VE"
        self.sigma_max = sigma_max
        self.alpha_fn = lambda t: torch.ones_like(t)
        self.alpha_T = 1
        self.rho_fn = lambda t: t
        self.rho_T = sigma_max
        self.dt_dlambda_fn = lambda t: -t
        
        self.lambda_inverse = lambda l: torch.exp(-l)
        
        self.f_fn = lambda t: torch.zeros_like(t)
        self.g2_fn = lambda t: 2 * t

    def get_timestep_from_lambda(self, lambda_t):
        return self.lambda_inverse(lambda_t.to(torch.float64))
    
    def get_alpha_prime(self, t):
        return torch.zeros_like(t).to(torch.float64)

    def get_sigma_prime(self, t):
        return torch.ones_like(t).to(torch.float64)

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t

class TrigFlowNoiseSchedule:
    def __init__(self, sigma_max=1.5707):
        self.name = "TrigFlow"
        self.sigma_max = sigma_max
        self.alpha_fn = lambda t: torch.cos(t)
        self.alpha_T = torch.DoubleTensor([sigma_max]).cos().item()
        self.rho_fn = lambda t: torch.tan(t)
        self.rho_T = torch.DoubleTensor([sigma_max]).tan().item()
        # self.lambda_fn = lambda t: -torch.log(self.rho_fn(t))
        # self.lambda_T = -torch.log(torch.DoubleTensor([self.rho_T]))
        
        self.f_fn = lambda t: -torch.tan(t)
        self.g2_fn = lambda t: 2 * torch.tan(t)

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_prime(self, t):
        return -torch.sin(t.to(torch.float64))

    def get_sigma_prime(self, t):
        return torch.cos(t.to(torch.float64))
    
    def get_abc(self, t):
        t = t.to(torch.float64)
        alpha_t, rho_t = self.alpha_fn(t), self.rho_fn(t)
        sigma_t = rho_t * alpha_t
        
        snr_T = 1 / self.rho_T ** 2
        snr_t = 1 / rho_t ** 2
        
        assert ((snr_T / snr_t) <= 1).all()
        a_t = (alpha_t / self.alpha_T) * (snr_T / snr_t)
        b_t = alpha_t * (1. - (snr_T / snr_t))
        c_t = sigma_t * torch.sqrt(1. - (snr_T / snr_t))
        return a_t, b_t, c_t
    
class TrigFlowVENoiseSchedule:
    def __init__(self, sigma_max=80.0):
        self.name = "TrigFlow_VE"
        self.sigma_max = sigma_max
        self.alpha_fn = lambda t: 1. / torch.sqrt(1. + torch.square(t))
        self.alpha_T = 1. / ((1. + (sigma_max**2))**0.5)
        self.rho_fn = lambda t: t
        self.rho_T = sigma_max
        
        self.f_fn = lambda t: -t / (1. + torch.square(t))
        self.g2_fn = lambda t: 2 * t / (1. + torch.square(t))

    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_prime(self, t):
        t = t.to(torch.float64)
        return - t / ( (1. + torch.square(t)) ** 1.5 )

    def get_sigma_prime(self, t):
        t = t.to(torch.float64)
        return 1 / ((1. + torch.square(t)) ** 1.5)
    
    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t
    
    def get_abc(self, t):
        t = t.to(torch.float64)
        alpha_t, rho_t = self.alpha_fn(t), self.rho_fn(t)
        sigma_t = rho_t * alpha_t
        
        snr_T = 1 / self.rho_T ** 2
        snr_t = 1 / rho_t ** 2
        
        assert ((snr_T / snr_t) <= 1).all()
        a_t = (alpha_t / self.alpha_T) * (snr_T / snr_t)
        b_t = alpha_t * (1. - (snr_T / snr_t))
        c_t = sigma_t * torch.sqrt(1. - (snr_T / snr_t))
        return a_t, b_t, c_t
    
    def get_abc_prime(self, t, a_t=None, b_t=None, c_t=None):
        alpha_T, rho_T = self.alpha_T, self.rho_T
        sigma_T = alpha_T * rho_T
        snr_T = 1 / rho_T ** 2
        
        alpha_t = self.alpha_fn(t)
        alpha_t_prime = self.get_alpha_prime(t)
        rho_t = self.rho_fn(t)
        sigma_t = alpha_t * rho_t
        sigma_t_prime = self.get_sigma_prime(t)
        
        if (a_t is None) or (b_t is None) or (c_t is None):
            a_t, b_t, c_t = self.get_abc(t)
        
        common = (2 * sigma_t_prime / sigma_t) - (alpha_t_prime / alpha_t)

        a_t_prime = a_t * common
        
        b_t_prime = common * snr_T * sigma_t.square() / alpha_t
        b_t_prime = alpha_t_prime - b_t_prime
        
        c_t_mult_c_t_prime = common * snr_T * (sigma_t ** 4) / (alpha_t ** 2)
        c_t_mult_c_t_prime = (sigma_t * sigma_t_prime) - c_t_mult_c_t_prime
        
        return a_t_prime, b_t_prime, c_t_mult_c_t_prime


class I2SBNoiseSchedule(NoiseSchedule):
    def __init__(self, n_timestep=1000, beta_min=0.1, beta_max=1.0):
        self.name = "I2SB"
        self.n_timestep, self.linear_start, self.linear_end = (
            n_timestep,
            beta_min / n_timestep,
            beta_max / n_timestep,
        )
        betas = (
            torch.linspace(
                self.linear_start**0.5,
                self.linear_end**0.5,
                n_timestep,
                dtype=torch.float64,
            ).cuda()
            ** 2
        )
        betas = torch.cat(
            [
                betas[: self.n_timestep // 2],
                torch.flip(betas[: self.n_timestep // 2], dims=(0,)),
            ]
        )
        std_fwd = torch.sqrt(torch.cumsum(betas, dim=0))
        std_bwd = torch.sqrt(torch.flip(torch.cumsum(torch.flip(betas, dims=(0,)), dim=0), dims=(0,)))

        self.alpha_fn = lambda t: torch.ones_like(t).float()
        self.alpha_T = 1
        self.rho_fn = lambda t: std_fwd[t]
        self.rho_T = std_fwd[-1]
        self.rho_bar_fn = lambda t: std_bwd[t]

        self.f_fn = lambda t: torch.zeros_like(t).float()
        self.g2_fn = lambda t: betas[t]
        
    def rho_inverse(self, rho_t):
        t = torch.argwhere(rho_t == self.rho_fn(torch.arange(self.n_timestep).to(rho_t.device))).squeeze()
        return t
    
    def get_f_g2(self, t):
        t = ((self.n_timestep - 1) * t).round().long()
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = ((self.n_timestep - 1) * t).round().long()
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = self.rho_bar_fn(t)
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t
    
    def get_timestep_from_lambda(self, lambda_t):
        rho_t = torch.exp(-lambda_t)
        t = self.rho_inverse(rho_t)
        return t

class PreCond:
    def __init__(self, ns):
        raise NotImplementedError

    def _get_scalings(self, t):
        raise NotImplementedError

    def _get_weightings(self, t, u):
        raise NotImplementedError
    
    def get_scalings(self, t, ndim):
        c_skip, c_in, c_out, c_noise = self._get_scalings(t)
        c_skip, c_in, c_out = [append_dims(item, ndim) for item in [c_skip, c_in, c_out]]
        return c_skip, c_in, c_out, c_noise
    
    def get_weightings(self, t, u, ndim):
        assert (t > u).all()
        t = t.to(torch.float64)
        u = u.to(torch.float64)
        return append_dims(self._get_weightings(t, u), ndim)
    
    def get_scalings_and_weightings(self, t, ndim):
        raise RuntimeError("get_scalings_and_weightings is deprecated, use get_scalings and get_weightings instead.")
        c_skip, c_in, c_out, c_noise = self._get_scalings(t)
        weightings = self._get_weightings(t)
        c_skip, c_in, c_out, weightings = [append_dims(item, ndim) for item in [c_skip, c_in, c_out, weightings]]
        return c_skip, c_in, c_out, c_noise, weightings


class I2SBPreCond(PreCond):
    def __init__(self, ns, n_timestep=1000, t0=1e-4, T=1.0):
        self.ns = ns
        self.n_timestep = n_timestep
        self.noise_levels = torch.linspace(t0, T, n_timestep).cuda() * n_timestep

    def _get_weightings(self, t, u):
        return 1 / (t - u)
    
    def _get_scalings(self, t):
        _, _, rho_t, _ = self.ns.get_alpha_rho(t)
        c_skip = torch.ones_like(t)
        c_in = torch.ones_like(t)
        c_out = -rho_t
        c_noise = self.noise_levels[((self.n_timestep - 1) * t).round().long()]
        return c_skip, c_in, c_out, c_noise


class DBMPreCond:
    def __init__(self, ns, sigma_data, cov_xy, sigma_data_end,
                 xT_normalization: bool, c_noise_type: str = 't'):
        self.ns, self.sigma_data, self.cov_xy = ns, sigma_data, cov_xy
        assert self.cov_xy == 0.0
        self.sigma_data_end = sigma_data
        self.c_noise_type = c_noise_type
        self.xT_normalization = xT_normalization
        assert self.c_noise_type in ['t', '10t', '1000t', '250logt']
    
    # def _get_weightings(self, t, m):
    #     return 1 / (t - m)
    
    # def get_weightings(self, t, m, ndim):
    #     assert (t > m).all()
    #     t = t.to(torch.float64)
    #     m = m.to(torch.float64)
    #     return append_dims(self._get_weightings(t, m), ndim)

    def _get_scalings(self, t):
        a_t, b_t, c_t = self.ns.get_abc(t)
        A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2 * a_t * b_t * self.cov_xy + c_t**2
        c_in = 1 / (A) ** 0.5
        c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy) / A
        c_out = (
            a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * c_t**2
        ) ** 0.5 * c_in
        
        if self.c_noise_type == 't':
            c_noise = t
        elif self.c_noise_type == '10t':
            c_noise = 10. * t
        elif self.c_noise_type == '1000t':
            c_noise = 1000. * t
        elif self.c_noise_type == '250logt':
            c_noise = 250. * torch.log(t + 1e-44)
        elif self.c_noise_type == 'pt25logt':
            c_noise = 0.25 * torch.log(t + 1e-44)
        else:
            raise NotImplementedError(f"c_noise_type '{self.c_noise_type}' not implemented.")
        
        c_in_xT = torch.ones_like(c_in)
        if self.xT_normalization:
            c_in_xT = c_in_xT / self.sigma_data_end
            
        return c_skip, c_in, c_out, c_noise, c_in_xT

    def get_scalings(self, t, ndim):
        t = t.to(torch.float64)
        c_skip, c_in, c_out, c_noise, c_in_xT = self._get_scalings(t)
        c_skip, c_in, c_out, c_in_xT = [append_dims(item, ndim) for item in [c_skip, c_in, c_out, c_in_xT]]
        return c_skip, c_in, c_out, c_noise, c_in_xT
        
    def _get_derivate_of_scalings(self, t):
        a, b, c = self.ns.get_abc(t)
        a_prime, b_prime, c_mult_c_prime = self.ns.get_abc_prime(t, a, b, c)
        # assert not c_mult_c_prime.isnan().any(), f"NaN detected in c_mult_c_prime calculation at t = {t[c_mult_c_prime.isnan()].unique()}"
        # assert not c_mult_c_prime.isinf().any(), f"Inf detected in c_mult_c_prime calculation at t = {t[c_mult_c_prime.isinf()].unique()}"
        
        A = a**2 * self.sigma_data_end**2 + b**2 * self.sigma_data**2 + 2 * a * b * self.cov_xy + c**2
        
        c_in = 1 / (A) ** 0.5
        # c_skip = (b * self.sigma_data**2 + a * self.cov_xy) / A
        c_out = (
            a**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * c**2
        ) ** 0.5 * c_in
        
        c_in_prime = (a * a_prime * (self.sigma_data_end**2)) + (b * b_prime * self.sigma_data**2) + (((a_prime * b) + (a * b_prime)) * self.cov_xy) + c_mult_c_prime
        # Nov 5: Wrong formula!
        # c_in_prime = -2 * c_in_prime * c_in.square() 
        # c_in_prime = -2 * c_in_prime / A
        # Nov 5: Corrected formula!
        c_in_prime = -c_in_prime * (c_in ** 3)
        
        c_out_prime = (a * a_prime * (((self.sigma_data_end * self.sigma_data)**2) - (self.cov_xy**2))) + ((self.sigma_data**2) * c_mult_c_prime)
        # c_out_prime = c_out_prime * c_in.square() / c_out
        c_out_prime = c_out_prime / (A * c_out)
        c_out_prime = c_out_prime + (c_out * c_in_prime / c_in)
        
        # c_skip_prime = (((self.sigma_data**2) * b_prime) + (self.cov_xy * a_prime)) * c_in.square()
        # c_skip_prime = c_skip_prime + (2 * c_skip * c_in_prime / c_in)
        c_skip_prime = (((self.sigma_data**2) * b_prime) + (self.cov_xy * a_prime)) / A
        # c_skip_prime = c_skip_prime + (2 * c_skip * c_in_prime / c_in)
        c_skip_prime = c_skip_prime + (2 * c_in * c_in_prime * (b * (self.sigma_data**2) + (a * self.cov_xy)))
        
        if self.c_noise_type == 't':
            c_noise_prime = torch.ones_like(t)
        elif self.c_noise_type == '10t':
            c_noise_prime = 10. * torch.ones_like(t)
        elif self.c_noise_type == '1000t':
            c_noise_prime = 1000. * torch.ones_like(t)
        elif self.c_noise_type == '250logt':
            c_noise_prime = 250. / (t + 1e-44)
        elif self.c_noise_type == 'pt25logt':
            c_noise_prime = 0.25 / (t + 1e-44)
        else:
            raise NotImplementedError(f"c_noise_type '{self.c_noise_type}' not implemented.")
        
        # assert not c_skip_prime.isnan().any(), f"NaN detected in c_skip_prime calculation at t = {t[c_skip_prime.isnan()].unique()}"
        # assert not c_in_prime.isnan().any(), f"NaN detected in c_in_prime calculation at t = {t[c_in_prime.isnan()].unique()}"
        # assert not c_out_prime.isnan().any(), f"NaN detected in c_out_prime calculation at t = {t[c_out_prime.isnan()].unique()}"
        # assert not c_skip_prime.isinf().any(), f"Inf detected in c_skip_prime calculation at t = {t[c_skip_prime.isinf()].unique()}"
        # assert not c_in_prime.isinf().any(), f"Inf detected in c_in_prime calculation at t = {t[c_in_prime.isinf()].unique()}"
        # assert not c_out_prime.isinf().any(), f"Inf detected in c_out_prime calculation at t = {t[c_out_prime.isinf()].unique()}"

        return c_skip_prime, c_in_prime, c_out_prime, c_noise_prime
    
    def get_derivate_of_scalings(self, t, ndim):
        t = t.to(torch.float64)
        c_skip_prime, c_in_prime, c_out_prime, c_noise_prime = self._get_derivate_of_scalings(t)
        c_skip_prime, c_in_prime, c_out_prime = [append_dims(item, ndim) for item in [c_skip_prime, c_in_prime, c_out_prime]]
        return c_skip_prime, c_in_prime, c_out_prime, c_noise_prime

class KarrasDenoiser:
    def __init__(
        self,
        noise_schedule,
        precond,
        t_max=1.0,
        t_min=0.0001,
        loss_norm="lpips",
    ):

        self.t_max = t_max
        self.t_min = t_min

        self.noise_schedule = noise_schedule
        self.precond = precond

        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")

    def bridge_sample(self, x0, xT, t, noise) -> torch.Tensor:
        a_t, b_t, c_t = [append_dims(item, x0.ndim).to(x0.dtype) for item in self.noise_schedule.get_abc(t)]
        samples = a_t * xT + b_t * x0 + c_t * noise
        return samples

    # def dbm_solver(self, xa, a, b, x0, xT, xa_noise):
    #     assert (b <= a).all()
        
    #     square_m1 = lambda first, second: ((first / second) ** 2) - 1.
        
    #     alpha_a, _, rho_a, _ = [append_dims(item, x0.ndim).to(x0.dtype) for item in self.noise_schedule.get_alpha_rho(a)]
    #     alpha_b, _, rho_b, _ = [append_dims(item, x0.ndim).to(x0.dtype) for item in self.noise_schedule.get_alpha_rho(b)]
    #     ratio_a, ratio_b = 1 / rho_a, 1 / rho_b
    #     alpha_T = self.noise_schedule.alpha_T
    #     ratio_T = 1 / self.noise_schedule.rho_T
        
    #     common_factor = torch.sqrt(square_m1(ratio_b, ratio_T) / square_m1(ratio_a, ratio_T))
    #     # assert not common_factor.isnan().any(), (square_m1(ratio_b, ratio_T) / square_m1(ratio_a, ratio_T) >= 0).all()
        
    #     first_a = alpha_b / alpha_a
    #     # assert not first_a.isnan().any(), f"NaN detected in dbm_solver output: timestep_a = {a.unique()}"
    #     # assert not first_a.isinf().any(), f"Inf detected in dbm_solver output: timestep_a = {a.unique()}"
    #     first_b = torch.square(ratio_a / ratio_b)
    #     # assert not first_b.isnan().any(), f"NaN detected in dbm_solver output: timestep_a = {a.unique()}"
    #     # assert not first_b.isinf().any(), f"Inf detected in dbm_solver output: timestep_a = {a.unique()}"
    #     first = xa * first_a * first_b * common_factor
        
    #     second_a = alpha_b
    #     # assert not second_a.isnan().any(), f"NaN detected in dbm_solver output: timestep_b = {b[second_a.isnan()].unique()}"
    #     # assert not second_a.isinf().any(), f"Inf detected in dbm_solver output: timestep_b = {b[second_a.isinf()].unique()}"
    #     second_b = 1. - torch.square(ratio_T / ratio_b)
    #     # assert not second_b.isnan().any(), f"NaN detected in dbm_solver output: timestep_b = {b[second_b.isnan()].unique()}"
    #     # assert not second_b.isinf().any(), f"Inf detected in dbm_solver output: timestep_b = {b[second_b.isinf()].unique()}"
    #     second_c = 1. - (1. / common_factor)
    #     # assert not second_c.isinf().any(), f"Inf detected in dbm_solver output: common_factor = {common_factor[second_c.isinf()].unique()} | ratio_b / ratio_T = {(ratio_b[second_c.isinf()].unique() / ratio_T)**2 - 1}"
    #     second = x0 * second_a * second_b * second_c
        
    #     third_a = alpha_b / alpha_T
    #     # assert not third_a.isnan().any(), f"NaN detected in dbm_solver output: timestep_b = {b[third_a.isnan()].unique()}"
    #     # assert not third_a.isinf().any(), f"Inf detected in dbm_solver output: timestep_b = {b[third_a.isinf()].unique()}"
    #     third_b = torch.square(ratio_T / ratio_b)
    #     # assert not third_b.isnan().any(), f"NaN detected in dbm_solver output: timestep_b = {b[third_b.isnan()].unique()}"
    #     # assert not third_b.isinf().any(), f"Inf detected in dbm_solver output: timestep_b = {b[third_b.isinf()].unique()}"
    #     third_c = 1. - common_factor
    #     # assert not third_c.isnan().any(), f"NaN detected in dbm_solver output: timestep_b = {(ratio_a)[third_c.isnan()].unique()}"
    #     # assert not third_c.isinf().any(), f"Inf detected in dbm_solver output: timestep_b = {(ratio_a)[third_c.isinf()].unique()}"
    #     third = xT * third_a * third_b * third_c
        
    #     xb = first + second + third
        
    #     # xb = torch.where(append_dims(a, x0.ndim) * torch.ones_like(x0) == torch.zeros_like(x0), x0, xb)
    #     # isnans = (append_dims(a, x0.ndim) * torch.ones_like(x0) == torch.zeros_like(x0))
    #     # xb[isnans] = x0[isnans]
    #     # assert not torch.isnan(xb).any(), f"NaN detected in dbm_solver output: timestep_a = {(append_dims(a, x0.ndim) * torch.ones_like(x0))[xb.isnan()].unique()} | t_max = {self.t_max}"
        
    #     a_eq_b = (a == b)
    #     xb[a_eq_b] = xa[a_eq_b]
        
    #     b_is_zero = (b == 0)
    #     xb[b_is_zero] = x0[b_is_zero]
        
    #     # Added on 24th Aug:
    #     _ratio_a = 1 / self.noise_schedule.get_alpha_rho(a)[2]
    #     a_eq_T = (_ratio_a == ratio_T)
    #     sigma_b = rho_b * alpha_b
    #     xb_a_eq_T = (xT * third_a * third_b) + (x0 * second_a * second_b) + (xa_noise * sigma_b * second_b.sqrt())
    #     xb[a_eq_T] = xb_a_eq_T[a_eq_T]
        
    #     assert not torch.isnan(xb).any()
    #     return xb

    def denoise(self, model, x_t, t, return_logvar: bool, **model_kwargs):
        assert not return_logvar
        c_skip, c_in, c_out, c_noise_t, c_in_xT = self.precond.get_scalings(t, x_t.ndim)
        
        weightings = 1. / c_out.square()
        assert not weightings.isnan().any(), f"NaN detected in weightings at t = {t[weightings.squeeze().isnan()].unique()}"
        
        y = model_kwargs["y"].clone() if "y" in model_kwargs else None
        xT = model_kwargs["xT"].clone()
        model_output, logvar = model(c_in * x_t, c_noise_t, xT=c_in_xT * xT, y=y, return_logvar=return_logvar)
        denoised = (c_out * model_output) + (c_skip * x_t)
        
        return model_output, denoised, logvar, weightings
    
    @torch.no_grad()
    def target_denoise(self, target_model, image, timestep, return_logvar, **model_kwargs):
        return self.denoise(target_model, image, timestep, return_logvar, **model_kwargs)[1]

    def get_dx_dt(self, x0, xT, t, xt) -> torch.Tensor:
        t = t.to(torch.float64)
        t = append_dims(t, x0.ndim)
        
        alpha_T, rho_T = self.precond.ns.alpha_T, self.precond.ns.rho_T
        snr_T = 1 / (rho_T ** 2)
        alpha_t, rho_t = self.precond.ns.alpha_fn(t), self.precond.ns.rho_fn(t)
        snr_t, sigma_t = 1 / (rho_t ** 2), alpha_t * rho_t
        
        first = (alpha_t / alpha_T) * (snr_T / snr_t) * xT
        second = alpha_t * (1. - (snr_T / snr_t)) * x0
        third_sq = sigma_t.square() * (1. - (snr_T / snr_t))
        
        # # 1103 Verification of a_t, b_t, c_t
        # a_t, b_t, c_t = self.noise_schedule.get_abc(t)
        # a_t_org = (alpha_t / alpha_T) * (snr_T / snr_t)
        # b_t_org = alpha_t * (1. - (snr_T / snr_t))
        # c_t_org = sigma_t.square() * (1. - (snr_T / snr_t))
        # assert torch.allclose(a_t, a_t_org), f"a_t mismatch: {a_t[a_t != a_t_org]} vs {a_t_org[a_t != a_t_org]}"
        # assert torch.allclose(b_t, b_t_org), f"b_t mismatch: {b_t[b_t != b_t_org]} vs {b_t_org[b_t != b_t_org]}"
        # assert torch.allclose(c_t.square(), c_t_org), f"c_t mismatch: {c_t.square()[c_t.square() != c_t_org]} vs {c_t_org[c_t.square() != c_t_org]}"

        f_t, g2_t = self.noise_schedule.get_f_g2(t)
        
        score_h = (first + second - xt) / third_sq
        score_d = ((xT * alpha_t / alpha_T) - xt) / ( (sigma_t ** 2) * ( (snr_t / snr_T) - 1. ) )
        dx_dt = (xt * f_t) - g2_t * ((score_h / 2) - score_d)
        
        # assert not dx_dt.isnan().any(), f"NaN detected in dx_dt calculation at t = {t.unique()}"
        # assert not dx_dt.isinf().any(), f"Inf detected in dx_dt calculation at t = {t.unique()}"
        return dx_dt
    
    def training_bridge_losses(self, model, x_start, t,  model_kwargs=None, noise=None, return_image_samples=False, ema_params=None, is_training=True, eps=1e-4):
        assert model_kwargs is not None
        assert noise is None
        # assert target_model is not None, "Target model must be provided for training losses."
        xT = model_kwargs.get("xT").clone()
        # xT = model_kwargs.pop("xT")
        mask = model_kwargs.pop("mask", None)
        assert mask is None
        disp_mask = model_kwargs.pop("disp_mask", None)
        padding_mask = model_kwargs.pop("padding_mask", None)
        loss_mask = None
        if disp_mask is not None:
            loss_mask = disp_mask
        if padding_mask is not None:
            loss_mask = padding_mask
        assert loss_mask is not None
        y = model_kwargs.get("y", None)
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        assert (t <= self.t_max).all()

        terms = {}
        
        x_t = self.bridge_sample(x_start, xT, t, noise)
        
        return_logvar = False
        _, denoised, logvar, weights = self.denoise(model, x_t, t, return_logvar=return_logvar, **model_kwargs)

        # 1. Pseudo-Huber loss
        loss_name = 'pseudo-huber'
        c = 0.00054 * math.sqrt(math.prod(x_start.shape[1:]))
        loss = torch.sqrt((denoised - x_start).square() + np.square(c)).sqrt() - c
        
        # 2. MSE loss
        # loss_name = 'mse'
        # loss = (denoised - x_start).square()
        
        # print(loss_mask.shape)
        # print(loss.shape)
        # print((loss_mask * loss).shape)
        # # print((loss_mask * loss)[0, 0, :19, :62].unique())
        # print((loss_mask * loss)[0, 0, :62, :19].unique())
        # print(mean_flat(loss_mask * loss))
        # exit()
        if loss_mask is not None:
            terms[f"xs_{loss_name}"] = mean_flat(loss_mask * loss)
            terms[loss_name] = mean_flat(weights * loss_mask * loss)
        else:
            terms[f"xs_{loss_name}"] = mean_flat(loss)
            terms[loss_name] = mean_flat(weights * loss)
        terms["loss"] = terms[loss_name]
        
        test_sigma_t = None
        test_x_ts = []
        
        ema_denoised_t = []
        ema_denoised_T = []
        
        if not is_training:
            assert ema_params is not None, "EMA parameters must be provided for evaluation."

            with torch.no_grad():
                ema_state_dict = deepcopy(model.state_dict())
                test_model = deepcopy(model).to(t.device)
                test_sigma_t = t[0] * torch.ones_like(t)
                
                test_noise = torch.randn_like(x_start)
                
                for idx, params in enumerate(ema_params):
                    for i, (name, _) in enumerate(model.named_parameters()):
                        assert name in ema_state_dict
                        ema_state_dict[name] = params[i]
                    
                    test_model.load_state_dict(ema_state_dict)
                    test_model.requires_grad_(False)
                    test_model.eval()
                    
                    ema_x_eps_T = self.target_denoise(test_model, xT, self.t_max * torch.ones_like(t), return_logvar, **model_kwargs).clamp(-1, 1)
                    if mask is not None:
                        ema_x_eps_T = ema_x_eps_T * mask + xT * (1 - mask)
                    
                    test_x_t = self.bridge_sample(ema_x_eps_T, xT, test_sigma_t, test_noise)
                    test_x_ts.append(test_x_t)
                    
                    ema_x_eps_xt = self.target_denoise(test_model, test_x_t, test_sigma_t, return_logvar, **model_kwargs).clamp(-1, 1)
                    if mask is not None:
                        ema_x_eps_xt = ema_x_eps_xt * mask + xT * (1 - mask)
                        
                    ema_denoised_t.append(ema_x_eps_xt)
                    ema_denoised_T.append(ema_x_eps_T)
                    
                    _loss = (ema_x_eps_xt - x_start).square()
                    _loss_T = (ema_x_eps_T - x_start).square()

                    terms[f"ema_idx{idx}_mse_t"] = mean_flat(_loss)
                    terms[f"ema_idx{idx}_mse_T"] = mean_flat(_loss_T)

            del test_model, ema_state_dict, ema_x_eps_xt, ema_x_eps_T
            torch.cuda.empty_cache()
            gc.collect()
                
        if return_image_samples:
            return terms, test_x_ts, denoised, test_sigma_t, ema_denoised_t, ema_denoised_T
        return terms


def karras_sample(
    diffusion,
    model,
    x_T,
    x_0,
    steps,
    prev_x_0=None,
    prev_x_0_prediction=None,
    mask=None,
    clip_denoised=True,
    model_kwargs=None,
    device=None,
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.0,
    eta=0.0,
    order=2,
    seed=None,
    dbm3_ratio_first=None,
    dbm3_ratio_second=None,
):
    prev_x_0 = None
    assert not x_0 is None 
    assert sampler in [
        "ground_truth", "euler", "exp_euler"
    ], "Only these samplers are supported currently."
    # print('diffusion.t_min', diffusion.t_min)
    # exit()
    
    if sampler == "ground_truth":
        gt = x_0.clamp(-1, 1)
        return (
        gt,
        [x_T],
        0,
        [gt],
        [diffusion.t_max],
        None,
    )
    
    if sampler in ["heun"]:
        ts = get_sigmas_karras(steps, diffusion.t_min, diffusion.t_max - 1e-4, rho, device=device)
    elif sampler in ['exp_euler', 'euler']:
        # ts = get_sigmas_karras(steps-1, diffusion.t_min, diffusion.t_max - 1e-4, device=device)
        ts = get_sigmas_uniform(steps-2, diffusion.t_min, diffusion.t_max - 1e-4, device=device)
    elif sampler in ["dbim_karras", "dbim_high_order_karras"]:
        ts = get_sigmas_karras_non0(steps, diffusion.t_min, diffusion.t_max - 1e-4, device=device)
    else:
        raise NotImplementedError(f"{sampler} has not been implemented for CM/CBM-based solvers just yet!")
        ts = get_sigmas_uniform_non0euler(steps, diffusion.t_min, diffusion.t_max - 1e-3, device=device)
        # ts = get_sigmas_uniform(steps, diffusion.t_min, diffusion.t_max - 1e-3, device=device)
    
    ts = ts.to(device)
    # print("Sampling timesteps:", ts, ts.shape)
    # exit()
    sample_fn = {
        "euler": sample_euler,
        "exp_euler": sample_exp_euler,
    }[sampler]

    sampler_args = dict(churn_step_ratio=churn_step_ratio, mask=mask, eta=eta, x0=x_0, order=order, seed=seed, 
                        prev_x_0=prev_x_0_prediction)
    
    if sampler == "cbm":
        sampler_args['x_0'] = x_0

    if sampler == "0euler_dbm3":
        dbm3_ratio_first, dbm3_ratio_second = float(dbm3_ratio_first), float(dbm3_ratio_second)
        assert dbm3_ratio_first > 0 and dbm3_ratio_first <= 1, f"dbm3_ratio_first (={dbm3_ratio_first}) must be between 0 and 1."
        assert dbm3_ratio_second >= dbm3_ratio_first and dbm3_ratio_second <= 1, f"dbm3_ratio_second (={dbm3_ratio_second}) must be between dbm3_ratio_first (={dbm3_ratio_first}) and 1."
        sampler_args['ratio_first'] = dbm3_ratio_first
        sampler_args['ratio_second'] = dbm3_ratio_second
    
    def denoiser(x_t, sigma):
        denoised = diffusion.denoise(model, x_t, sigma, return_logvar=False, **model_kwargs)[1]
        
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0, patorch, nfe, pred_x0, sigmas, noise = sample_fn(
        denoiser,
        diffusion,
        x_T,
        ts,
        **sampler_args,
    )
    if dist.get_rank() == 0:
        print("nfe:", nfe)

    return (
        x_0.clamp(-1, 1),
        [x.clamp(-1, 1) for x in patorch],
        nfe,
        [x.clamp(-1, 1) for x in pred_x0],
        sigmas,
        noise,
    )


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def get_sigmas_karras_non0(n, t_min, t_max, rho=7.0, device="cpu"):
    ramp = torch.linspace(0, 1, n + 1)
    min_inv_rho = t_min ** (1 / rho)
    max_inv_rho = t_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas.to(device)

def get_sigmas_uniform_non0euler(n, t_min, t_max, device="cpu"):
    return torch.linspace(t_max, t_min, n + 1).to(device)

def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    # return torch.linspace(t_max, t_min, n + 1).to(device)
    return append_zero(torch.linspace(t_max, t_min, n + 1)).to(device)

@torch.no_grad()
def sample_euler(
    denoiser,
    diffusion,
    x,
    sigmas,
    callback=None,
    mask=None,
    seed=None,
    prev_x_0=None,
    **kwargs,
):
    """Implements Algoritorchm 2 (Heun steps) from Karras et al. (2022)."""
    assert mask is None
    use_prev_x0 = not (prev_x_0 is None)
    # num_of_steps_to_use_prev_x0_for = (len(indices) - 1) // 2
    num_of_steps_to_use_prev_x0_for = 4
    # x_T = x
    x_T = x.clone()
    path = [x.detach().clone().cpu()]
    pred_x0 = []

    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))
    
    noise = torch.randn_like(x)

    nfe = 0
    if use_prev_x0 and (num_of_steps_to_use_prev_x0_for > 0):
        print('Using the given prev_x_0 for first', num_of_steps_to_use_prev_x0_for, 'step(s).')
        x0_hat = prev_x_0.clone()
        assert not torch.isnan(x0_hat).any()
    else:
        x0_hat = denoiser(x, diffusion.t_max * s_in)
        assert not torch.isnan(x0_hat).any()
        
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    
    if seed is None:
        noise = torch.randn_like(x)
    else:
        generator = BatchedSeedGenerator(seed)
        noise = generator.randn_like(x)
    first_noise = noise
    
    a_t, b_t, c_t = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_abc(sigmas[0] * s_in)]
    x = a_t * x_T + b_t * x0_hat + c_t * first_noise
    
    # print('initial sde:', 'from', diffusion.t_max, 'to', sigmas[0].unique().item())
    # print('a_t', a_t.unique().item(), 'b_t', b_t.unique().item(), 'c_t', c_t.unique().item())
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1
    
    assert mask is None
    
    for j, i in enumerate(indices):
        
        sigma_hat = sigmas[i]
        sigma_next = sigmas[i + 1]
        # is_deterministic_step = (sigma_next == 0)
        # is_deterministic_step = True
        is_deterministic_step = False
        
        # print(i, 'from', sigma_hat.unique().item(), 'to', sigma_next.unique().item())
        
        nfe += 1
        
        dt = sigma_next - sigma_hat
        f_t, g2_t = [ append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_f_g2(sigma_hat * s_in) ]
        a_t, b_t, c_t = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_abc(sigma_hat * s_in)]
        
        if use_prev_x0 and (i < num_of_steps_to_use_prev_x0_for):
            print('Using the given prev_x_0_prediction for step:', i)
            denoised = prev_x_0.clone()
        else:
            denoised = denoiser(x, sigma_hat * s_in)
            assert not torch.isnan(denoised).any()
        
        alpha_t, alpha_bar_t, _, rho_bar_t = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigma_hat * s_in)]
        
        grad_logq = -(x - (a_t * x_T + b_t * denoised)) / c_t**2
        assert not torch.isnan(grad_logq).any()
        
        grad_logpxTlxt = -(x - alpha_bar_t * x_T) / (alpha_t**2 * rho_bar_t**2)
        assert not torch.isnan(grad_logpxTlxt).any()
        
        d = f_t * x - g2_t * ((0.5 if is_deterministic_step else 1) * grad_logq - grad_logpxTlxt)
        assert not torch.isnan(d).any()
        
        x = x + (d * dt) + (0 if is_deterministic_step else 1) * torch.randn_like(x) * ((dt).abs() ** 0.5) * g2_t.sqrt()
        assert not x.isnan().any()
        
        path.append(x.detach().cpu())
        pred_x0.append(denoised.detach().cpu())
        
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )

    return x, path, nfe, pred_x0, sigmas, None


@torch.no_grad()
def sample_exp_euler(
    denoiser,
    diffusion,
    x,
    sigmas,
    order,
    callback=None,
    mask=None,
    seed=None,
    prev_x_0=None,
    **kwargs,
):
    """Implements Algoritorchm 2 (Heun steps) from Karras et al. (2022)."""
    assert mask is None
    assert order in [1, 2]
    use_prev_x0 = not (prev_x_0 is None)
    # num_of_steps_to_use_prev_x0_for = (len(indices) - 1) // 2
    num_of_steps_to_use_prev_x0_for = 4
    # x_T = x
    x_T = x.clone()
    path = [x.detach().clone().cpu()]
    pred_x0 = []

    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))
    
    noise = torch.randn_like(x)

    nfe = 0
    if use_prev_x0 and (num_of_steps_to_use_prev_x0_for > 0):
        print('Using the given prev_x_0_prediction for first', num_of_steps_to_use_prev_x0_for, 'step(s).')
        x0_hat = prev_x_0.clone()
        assert not torch.isnan(x0_hat).any()
    else:
        print('Not using prev_x_0_prediction!')
        x0_hat = denoiser(x, diffusion.t_max * s_in)
        assert not torch.isnan(x0_hat).any()
    
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    
    if seed is None:
        noise = torch.randn_like(x)
    else:
        generator = BatchedSeedGenerator(seed)
        noise = generator.randn_like(x)
    first_noise = noise
    
    a_t, b_t, c_t = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_abc(sigmas[0] * s_in)]
    x = a_t * x_T + b_t * x0_hat + c_t * first_noise
    
    # print('initial sde:', 'from', diffusion.t_max, 'to', sigmas[0].unique().item())
    # print('a_t', a_t.unique().item(), 'b_t', b_t.unique().item(), 'c_t', c_t.unique().item())
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1
    
    snr_T = 1 / (diffusion.noise_schedule.rho_T ** 2)
    alp_T = diffusion.noise_schedule.alpha_T
    square_m1 = lambda first, second: ((first / second) ** 2) - 1.
    
    assert mask is None
    
    for j, i in enumerate(indices):
        
        sigma_hat = sigmas[i]
        sigma_next = sigmas[i + 1]
        is_last_step = (sigma_next == 0).all()
        # is_stochastic = (sigma_next == 0).all()
        # is_stochastic = True
        
        # print(i, 'from', sigma_hat.unique().item(), 'to', sigma_next.unique().item())
        
        nfe += 1
        if use_prev_x0 and (i < num_of_steps_to_use_prev_x0_for):
            print('Using the given prev_x_0_prediction for step:', i)
            denoised = prev_x_0.clone()
        else:
            denoised = denoiser(x, sigma_hat * s_in)
            assert not torch.isnan(denoised).any()
        
        # if is_stochastic:
            
        if (order == 1) or is_last_step:
            a_t, b_t, c_t = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_abc(sigma_next * s_in)]
            x = a_t * x_T + b_t * denoised + c_t * torch.randn_like(x)
            
            if is_last_step:
                assert (a_t == 0).all()
                assert (c_t == 0).all()
        else:
            assert order == 2
            nfe += 1
            
            ratio = 0.5
            sigma_mid = sigma_hat + ratio * (sigma_next - sigma_hat)
            
            alp_hat, _, rho_hat, __ = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigma_hat * s_in)]
            alp_mid, _, rho_mid, __ = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigma_mid * s_in)]
            alp_next, _, rho_next, __ = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigma_next * s_in)]
            
            a_mid, b_mid, c_mid = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_abc(sigma_mid * s_in)]
            a_next, b_next, c_next = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_abc(sigma_next * s_in)]

            x_mid = a_mid * x_T + b_mid * denoised + c_mid * torch.randn_like(x)
            assert not torch.isnan(x_mid).any()
            
            if use_prev_x0 and (i < num_of_steps_to_use_prev_x0_for):
                print('-- Using the given prev_x_0_prediction for denoised_mid for step:', i)
                denoised_mid = prev_x_0.clone()
            else:
                denoised_mid = denoiser(x_mid, sigma_mid * s_in)
                assert not torch.isnan(denoised_mid).any()
            
            snr_hat = 1 / (rho_hat ** 2)
            snr_mid = 1 / (rho_mid ** 2)
            snr_next = 1 / (rho_next ** 2)
            
            lambda_hat = -torch.log(rho_hat)
            lambda_mid = -torch.log(rho_mid)
            lambda_next = -torch.log(rho_next)
            
            sig_next = alp_next * rho_next

            first = (snr_hat / snr_next) * (alp_next / alp_hat)
            second = alp_next * (1. - (snr_hat / snr_next))
            third = alp_next * (lambda_next - lambda_hat + (snr_hat / snr_next) - 1.)
            fourth = sig_next * torch.sqrt(1. - (snr_hat / snr_next))
            
            if use_prev_x0 and (i <= num_of_steps_to_use_prev_x0_for):
                derivative = 0.
            else:
                derivative = (denoised_mid - denoised) / (lambda_mid - lambda_hat)    
            
            x = first * x + second * denoised + third * derivative + fourth * torch.randn_like(x)
        # else:
            
        #     alp_hat, _, rho_hat, __ = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigma_hat * s_in)]
        #     alp_next, _, rho_next, __ = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigma_next * s_in)]
            
        #     # sig_hat = alp_hat * rho_hat
        #     # sig_next = alp_next * rho_next
            
        #     snr_hat = 1 / (rho_hat ** 2)
        #     snr_next = 1 / (rho_next ** 2)
            
        #     common = torch.sqrt(square_m1(snr_next, snr_T) / square_m1(snr_hat, snr_T))
            
        #     first = (alp_next / alp_hat) * (snr_hat / snr_next) * common
        #     second = (alp_next / alp_T) * (snr_T / snr_next) * (1. - common)
        #     third = alp_next * (snr_T / snr_next) * square_m1(snr_next, snr_T) * (1. - (1. / common))
            
        #     x = first * x + second * x_T + third * denoised
        
        assert not x.isnan().any()
        
        path.append(x.detach().cpu())
        pred_x0.append(denoised.detach().cpu())
        
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )

    return x, path, nfe, pred_x0, sigmas, None
