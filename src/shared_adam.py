"""A3C Visual Navigation Mobile Robot project.

Create and initialize the shared optimizer (ADAM), 

@Author: Emanuel-Ionut Otel
@University: National Chung Cheng University
@Created: 2022-06-13
@Links: www.ccu.edu.tw
@Contact: manuotel@gmail.com
"""

#### ---- IMPORTS AREA ---- ####
import torch
#### ---- IMPORTS AREA ---- ####

class SharedAdam(torch.optim.Adam):
    """Extends ADAM otimizer for shared states.
    
    :attr params: iterable of parameters to optimize or dicts defining parameter groups
    :attr lr: learning rate (default: 1e-3)
    :attr betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    :attr eps: term added to the denominator to improve numerical stability (default: 1e-8)
    :attr weight_decay: weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=7e-4, betas=(0.92, 0.99), eps=1e-8, weight_decay=0):
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
