"""
Create and initialize the shared optimizer (ADAM), 
the parameters in the optimizer will shared in the multiprocessors.

In other words this optimizer will ensure that the master A3C nerual network will
learn from all the A2C workers.
"""

import torch
import numpy as np


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=7e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        #### State initialization ####
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                #### Share in memory ####
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class SharedAdam2(torch.optim.Adam):
    
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        
        super(SharedAdam2, self).__init__(params, lr, betas, eps, weight_decay)
        
        # init to 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
    
        # share adam's param
        def share_memory(self):
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state['step'].share_memory_()
                    state['exp_avg'].share_memory_()
                    state['exp_avg_sq'].share_memory_()
        
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

