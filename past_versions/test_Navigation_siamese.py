import torch
import torch.nn as nn
from torchvision import transforms
from utils import v_wrap, set_init, push_and_pull, record
from torch.autograd import Variable
import torch.nn.functional as F
from Ai2thor_Env_Setup import ActiveVisionDatasetEnv

from shared_adam import SharedAdam
import os
import sys
###
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from collections import Counter
import random

# from train_Navigation_1scene import Global_Net, set_all_seeds, tensor_to_RGB, CNN_Encoder_setup, make_env
# from train_Navigation_next_frame_1scene import Global_Net, set_all_seeds, tensor_to_RGB, CNN_Encoder_setup, make_env
# from train_Navi_siamese_VAE_next_frame import Global_Net, set_all_seeds, tensor_to_RGB, CNN_Encoder_setup, make_env
from ttrain_Navi_VAE_recon import Global_Net, set_all_seeds, tensor_to_RGB, CNN_Encoder_setup, make_env

# from VAE_NN import VAE
# from VAE_NN_256 import VAE
# from VAE256_siamese import VAE
from CBAM_VAE import VAE
# from CNN_Encoder_256 import CNN_Encoder
##
unloader = transforms.ToPILImage()
##
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, Manager, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
##
from torch.utils.tensorboard import SummaryWriter
os.chdir('..')
PATH_to_log_dir = './Testing/my_runs/FloorPlan303_SACA_Modify_5act_old_reward_Navigation_testing_First100ep'
writer = SummaryWriter(PATH_to_log_dir)

# Device
CUDA_DEVICE_NUM = 0
DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)


def CNN_Encoder_setup_testing(Is_VAE, Weight_DIR, z_dim):
# Load VAE model weight
    if Is_VAE:
        model_Encoder = VAE(z_dim)
        model_Encoder.to(DEVICE)
        CNN_filename = 'VAE'              # name for save model
        model_A3C = Global_Net(5)         # N_A = 3
        # model_A3C = Global_Net(5)            # N_A = 3->5
        model_A3C.to(DEVICE)
        if DEVICE == 'cpu':
            VAE_state_dict = torch.load(Weight_DIR[0], map_location=lambda storage, loc: storage)
            A3C_state_dict = torch.load(Weight_DIR[1], map_location=lambda storage, loc: storage)
        else:
            VAE_state_dict = torch.load(Weight_DIR[0], map_location=lambda storage, loc: storage.cuda(CUDA_DEVICE_NUM))
            A3C_state_dict = torch.load(Weight_DIR[1], map_location=lambda storage, loc: storage.cuda(CUDA_DEVICE_NUM))
        model_Encoder.load_state_dict(VAE_state_dict)
        model_A3C.load_state_dict(A3C_state_dict)


# Load A3C & Encoder model weight
#     else:
#         model_Encoder = CNN_Encoder()
#         model_Encoder.to(DEVICE)
#         CNN_filename = 'Encoder'          # name for save model
#         model_A3C = Global_Net(3)         # N_A = 3
#         model_A3C.to(DEVICE)
#         if DEVICE == 'cpu':
#             Encoder_A3C_state_dict = torch.load(Weight_DIR, map_location=lambda storage, loc: storage)
#         else:
#             Encoder_A3C_state_dict = torch.load(Weight_DIR, map_location=lambda storage, loc: storage.cuda(CUDA_DEVICE_NUM))
#         model_Encoder.load_state_dict(Encoder_A3C_state_dict['Encoder256.state_dict'])
#         model_A3C.load_state_dict(Encoder_A3C_state_dict['gnet.state_dict'])

    model_Encoder.eval()
    model_A3C.eval()
    for p in model_Encoder.parameters():  # 固定Encoder參數
        p.requires_grad = False
    for p in model_A3C.parameters():  # 固定A3C參數
        p.requires_grad = False

    return model_Encoder, CNN_filename, model_A3C


def record(collide_ep, succeed, ep_spl, ep_act_counter):
    ep_collide_list.append(collide_ep)
    succeed_list.append(succeed)
    Ep_spl.append(ep_spl)
    # Act_Count += ep_act_counter


if __name__ == '__main__':
# Env Setup
#     all_random_seed = 543
#     set_all_seeds(all_random_seed)
    # Set_resize = (128, 128)                   # 原圖 300x300
    Set_resize = (256, 256)                     # 原圖 300x300
    VAE_z_dim = 64

    L_5 = False
    MAX_task_step = 1000                        # 固定每個 Task 最大的 step 數
    Self_Adjust_task_step_N = 40    # N           # 自動調整每一個task的最大 step 數  (不是None則預設為True)
    testing_times_per_target = 100               # 每個target test 次數

# Model Setup
    Is_VAE = True

    if Is_VAE:
        # VAE+A3C
        # Weight_DIR = ['./VAE_recon_weight/44.Beta0.5VAE_recon_64z_4-256_batch256.pt', './n_A3Cweight/n8.Beta0.5-A3C_VAE256-1_recon_100N_ep500_.pt']
        Weight_DIR = ['./VAE_recon_weight/48.Beta2.5VAE_recon_64z_4-256_batch256.pt', './n_A3Cweight/Beta/n30.Beta2.5-A3C_VAE256-1_recon_100N_ep500_.pt']
        # Weight_DIR = ['./VAE_recon_weight/35.VAE_recon_64z_4-256_batch256.pt', './n_A3Cweight/n17.A3C_VAE256-1_recon_100N_ep500_.pt']
        # Weight_DIR = ['./VAE_recon_weight/35.VAE_recon_64z_4-256_batch256.pt', './n_A3Cweight/n12.A3C_VAE256-1_recon_100N_ep500_.pt']
    # Final version
        #Fl303
        # Weight_DIR = ['./VAE_recon_weight/35.VAE_recon_64z_4-256_batch256.pt', './n_A3Cweight/Final/Final_40.Fl303_all_randA3C_VAE256-1_recon_100N_ep500_.pt']
        Weight_DIR = [os.path.dirname(__file__)+'/6.SA_CA_VAE_recon_64z_4-256_batch256.pt', os.path.dirname(__file__)+'/TEST_3act_Floor303_1Task_SA_CA_NewVer_A3C_VAE256-1_recon_20N_ep5000_.pt']

    # else:
    #     # Encoder&A3C
    #     # Weight_DIR = './A3C_weight/修改reward錯誤後/調整task 平均設置後(100N)/10N or 50N/50N/A3C&Encoder256-1_ep500_.pt'
    #     Weight_DIR = './A3C_weight/FloorPlan3/FloorPlan3_A3C&Encoder256-1_ep1000_.pt'

    model_Encoder, CNN_filename, model_A3C = CNN_Encoder_setup_testing(Is_VAE, Weight_DIR, VAE_z_dim)  # VAE or traditional CNN Encoder
    print('CNN model: ', CNN_filename + str(Set_resize[0]))

    for name, param in model_Encoder.named_parameters():  # print出model_Encoder可/不可被更新的 NN layers weight (param.requires_grad=True or False)
        if param.requires_grad:
            print('%s training' % name)
        else:
            print('%s freeze ' % name)
    for name, param in model_A3C.named_parameters():
        if param.requires_grad:
            print('%s training' % name)
        else:
            print('%s freeze ' % name)

##### testing ########
    env = ActiveVisionDatasetEnv()
    k = 303                               # kitchen:1~30, Living roon:201~230, Bedroom:301~330, Bathroom:401~430
    # k = 3                               # kitchen:1~30, Living roon:201~230, Bedroom:301~330, Bathroom:401~430
    _cur_world = 'FloorPlan' + str(k)   # 決定sence
    print(_cur_world)
    all_goal_info = env.setup_testing_env(_cur_world)
    # all_goal_info = [all_goal_info[1]]   # 1 target : Microwave (FloorPlan3)
    # all_goal_info = [all_goal_info[-1]]   # 1 target : LightSwitch (FloorPlan206)

    target_record = []
    ep_collide_list = []
    succeed_list = []
    total_collide_cnt = 0
    Ep_spl = []
    Act_Count = Counter()                       # 計數所有 ACTION(3) 執行次數
    Overall_ep = 1000

    start_time = time.time()                    # 開始計時
    # fig_diplay_target = plt.figure()
    # fig_diplay_target, ax1 = plt.subplots()
    # fig_diplay_target.canvas.manager.window.wm_geometry('+100+100')

    for target in all_goal_info:
        print('Target: ', target)
        target_cur_Ep = 0  # 本次target的當前testing次數的第幾次
        for i in range(testing_times_per_target):
            Ep_step = 1         # 當前Ep執行第幾次step
            collide_ep = 0
            ep_act = []
            Ep_start_time = time.time()

            cur_obs, target_obs, shortest_len, pre_action, _, target_name = env.reset_for_testing(target, MAX_task_step, Self_Adjust_task_step_N, L_5)  # pre_action t size 3->5
            target_record.append(target_name)
            # plt.ion()
            # # ax1 = fig_diplay_target.add_subplot(1, 1, 1)
            # ax1.imshow(target_obs)
            # plt.ioff()
            # plt.pause(0.01)
            cur_obs = torch.Tensor(cv2.resize(cur_obs.numpy(), Set_resize)).transpose(0, 2)  # [300,300,3] -> [3,128,128]
            target_obs = torch.Tensor(cv2.resize(target_obs.numpy(), Set_resize)).transpose(0, 2)
            cur_obs_ = cur_obs[None, :].to(DEVICE)  # [3,128,128] -> [1,3,128,128]
            target_obs_ = target_obs[None, :].to(DEVICE)
            pre_action_ = pre_action[None, :].to(DEVICE)  # 6 -> [1, 6]
            while True:
                # input_state, recon_pre_obs = model_Encoder.Navigation_forward(cur_obs_, target_obs_)  #  VAE encode and Merge cur & target
                input_state = model_Encoder.Navigation_forward(cur_obs_, target_obs_)  #  VAE encode and Merge cur & target
                                                                                                                                        # if CNN_Encoder: recon_pre_obs = None
                if 'collided' not in locals():      # 2022/07/19 add last_collide(collided)
                    collided = False
                action = model_A3C.choose_act(input_state, pre_action_, collided)

                new_obs, collided, _, done, succeed = env.step(action,1,0.75)   # 從環境中 get state, reward ,done or not
                # if collided == True:
                #     new_obs, collided, _, done, succeed = env.step(1)  # 從環境中 get state, reward ,done or not
                #     # new_obs, collided, _, done, succeed = env.step(3)  # 從環境中 get state, reward ,done or not
                # else:
                #     new_obs, collided, _, done, succeed = env.step(action)  # 從環境中 get state, reward ,done or not
                
                ep_act.append(action)
                # print('action: ', action)
                if collided:
                    collide_ep += 1
                
                act_onehot = torch.zeros(pre_action_.shape)  # shape [1, 3]
                cur_obs_ = torch.Tensor(cv2.resize(new_obs.numpy(), Set_resize)).transpose(0, 2)[None, :]
                act_onehot[0, action] = 1.

                cur_obs_ = cur_obs_.to(DEVICE)  # [1,3,128,128]
                pre_action_ = act_onehot.to(DEVICE)  # [1, 3]

                if done:                                                        # done and print information
                    print('----------------Done------------------')
                    if succeed:
                        print('==== Succeed ====')
                        ep_spl = shortest_len/max(Ep_step, shortest_len)     # 計算單次Ep的SPL

                    else:
                        print('==== Fail ====')
                        ep_spl = 0

                    writer.add_scalar('Ep_SPL / Overall_ep', ep_spl, Overall_ep)
                    ep_act_counter = Counter(ep_act)                           # 計數 ACTION(3) 個別執行次數
                    total_collide_cnt += collide_ep                         # 計數所有碰撞數
                    record(collide_ep, succeed, ep_spl, ep_act_counter)
                    print('collide_ep: ', collide_ep)
                    print('ep_spl: ', ep_spl)
                    print('ep_act: ', ep_act)
                    print('act_cnt: ', sorted(ep_act_counter.items()))
                    print('Ep Time elapsed: %.2f sec' % ((time.time() - Ep_start_time)))
                    time.sleep(3)
                    break

                Ep_step += 1
            target_cur_Ep += 1
            Overall_ep += 1
            print(target[0], ' target_cur_Ep  %s' % target_cur_Ep, '/%s' % testing_times_per_target)

    print('========================== Finish All Target Navigation ===============================')
    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

# Analysis and Statistics
    total_collide_cnt = sum(ep_collide_list)  # Collision
    succeed_num = succeed_list.count(True)  # SR
    succeed_rate = succeed_num * 100 / len(succeed_list)
    SPL = sum(Ep_spl) / len(Ep_spl)  # final SPL

    print('Action total: ', sorted(Act_Count.items()))       # show all number of actions
    # print('Global all_act Length: ', len(global_all_act))
    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    print('Collision_ep: ', ep_collide_list)
    print('Total Num of collision :', total_collide_cnt)
    print('Success Rate: %.2f%%' % succeed_rate)
    print('SPL: ', SPL)
    print('Using VAE: ', Is_VAE)


    # plt.pause(0)
    plt.show()