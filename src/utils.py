"""A3C Visual Navigation Mobile Robot project.

This is a utilities script.

@Author: Emanuel-Ionut Otel
@University: National Chung Cheng University
@Created: 2022-06-13
@Links: www.ccu.edu.tw
@Contact: manuotel@gmail.com
"""

#### ---- IMPORTS AREA ---- ####
import torch, gc
import numpy as np
from torch import nn
#### ---- IMPORTS AREA ---- ####


def tensor_to_RGB(tensor) -> np.ndarray:
    """This function is used to convert a tensor to RGB.

    :param tensor: The tensor to be converted.

    return: The converted tensor.
    """
    image = tensor.cpu().clone()
    if image.dim()>3:
        image = image.squeeze(0)
    image = unloader(image)
    image = np.asarray(image).transpose(1,0,2)
    return image


def v_wrap(np_array, dtype=np.float32) -> torch.Tensor:
    """This function is used to convert a numpy array to a tensor.

    :param np_array: The numpy array to be wrapped.
    :param dtype: The data type of the tensor.

    return: The wrapped tensor.
    """
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def init_layer(layers) -> None:
    """This function is used to initialize the layers with 0 mean and 0.1 standard deviation.

    :param layers: The layers to be initialized.

    return: None
    """
    for layer in layers:
        try:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)   # 設置bias (起始為0）
        except:
            pass


def push_and_pull(optimizer, 
                  local_net, 
                  global_net, 
                  buffer_cur_z, 
                  buffer_target_z, 
                  buffer_pre_action, 
                  buffer_action, 
                  buffer_reward, 
                  gamma, 
                  DEVICE, 
                  hx, 
                  cx) -> None:
    """This function is used to push and pull the gradients.

    :param optimizer: The optimizer used to optimize the gradients.
    :param local_net: The local network. (A2C)
    :param global_net: The global network. (A3C)
    :param buffer_cur_z: The buffer used to store the current z.
    :param buffer_target_z: The buffer used to store the target z.
    :param buffer_pre_action: The buffer used to store the previous action.
    :param buffer_action: The buffer used to store the action.
    :param buffer_reward: The buffer used to store the reward.
    :param gamma: The discount factor.
    :param DEVICE: The device used to train the model.
    :param hx: The hidden state.
    :param cx: The cell state.
    
    return: None
    """
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
    """This function is used to record the results.

    :param res_queue: The queue used to store the results.
    :param ep_r: The reward of the episode.
    :param succeed: The success of the episode.
    :param ep_spl: The SPL of the episode.

    :return: None
    """
    res_queue.put([ep_r, succeed, ep_spl])


def save_name_file(prefix:str='', 
                   model_filename:str='', 
                   input_size:int=0, 
                   max_glob_ep:int=0, 
                   self_adjust_taxk_setp_n=None) -> str:
    """This function is used to save the model.

    :param prefix: The prefix of the model name.
    :param model_filename: The model name.
    :param input_size: The input size of the model.
    :param max_glob_ep: The maximum number of global episodes.
    :param self_adjust_taxk_setp_n: The number of self-adjusted task steps.

    :return: The model name.
    """
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