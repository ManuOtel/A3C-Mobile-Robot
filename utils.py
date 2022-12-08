"""
Functions that use multiple times
"""
import warnings
warnings.filterwarnings("ignore")
from torch import nn
import torch, gc
import numpy as np
from collections import Counter
import time


def tensor_to_RGB(tensor):
    image = tensor.cpu().clone()
    if image.dim()>3:
        image = image.squeeze(0)
    image = unloader(image)
    image = np.asarray(image).transpose(1,0,2)
    return image


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def init_layer(layers):
    for layer in layers:
        try:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)   # 設置bias (起始為0）
        except:
            pass

def push_and_pull(optimizer, local_net, global_net, buffer_cur_z, buffer_target_z, buffer_pre_action, buffer_action, buffer_reward, gamma, DEVICE, hx, cx):
    value = 0

    buffer_v_target = []
    for reward in buffer_reward[::-1]:
        value = reward + gamma * value
        buffer_v_target.append(value)
    buffer_v_target.reverse()

    loss = local_net.loss_func(
        torch.vstack(buffer_cur_z),
        torch.vstack(buffer_target_z),
        torch.vstack(buffer_pre_action),
        hx,
        cx,
        torch.tensor(buffer_action, device=DEVICE),
        torch.tensor(buffer_v_target, device=DEVICE)[:, None]
        )

    optimizer.zero_grad()
    loss.backward()
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        gp._grad = lp.grad
    optimizer.step()

    local_net.load_state_dict(global_net.state_dict())
    gc.collect()


def record(res_queue, ep_r, succeed, ep_spl):
    res_queue.put([ep_r, succeed, ep_spl])


def save_name_file(prefix='', model_filename='', input_size=0, max_glob_ep=0, self_adjust_taxk_setp_n=None):
    # A3C+Encoder
    if model_filename == 'Encoder':
        if self_adjust_taxk_setp_n is not None:
            Save_A3C_model_filename = 'A3C&' + model_filename + str(input_size)\
                                      + '-1_'+str(self_adjust_taxk_setp_n)+'N_ep' \
                                      + str(max_glob_ep)
        else:
            Save_A3C_model_filename = 'A3C&' + model_filename + str(input_size) + '-1_ep' + str(max_glob_ep) 
        # Save_A3C_model_filename = 'FloorPlan3_A3C&' + model_filename + str(Set_resize[0]) + '-1_ep' + str(MAX_Global_Ep) + '_.pt'

    # A3C (based on VAE)
    else:
        if self_adjust_taxk_setp_n is not None:
            Save_A3C_model_filename = prefix + 'A3C_' + model_filename + str(input_size) \
                                      + '-1_recon_'+str(self_adjust_taxk_setp_n)+'N_ep' \
                                      + str(max_glob_ep)
        else:
            Save_A3C_model_filename = 'A3C_' + model_filename + str(input_size) + '-1_ep' + str(max_glob_ep) 
    gc.collect()
    return Save_A3C_model_filename

if __name__=='__main__':
    print(save_name_file('prefix','test', 1, 1, 1))