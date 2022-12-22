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

ACTIONS = ['RotateRight', 'RotateLeft', 'MoveAhead', 'RotateRight90', 'RotateLeft90']

# PATH_to_log_dir = './my_runs/Navigation/Single_Task/Small_Rotation_Test/FloorPlan3_1Task_SA_CA_45D_R2G_VAE256_recon_64'
# PATH_to_log_dir = './my_runs/Navigation/Single_Task/NewVer_500_episode/FloorPlan3_1Task_SA_CA_VAE256_recon_64_2nd'
# PATH_to_log_dir = './my_runs/Navigation/Single_Task/Small_Rotation_Test/FloorPlan303_1Task_VAE256_recon_64'
PATH_to_log_dir = './my_runs/Navigation/Single_Task/NewVer_500_episode/Modify_5act_New_reward_FloorPlan303_1Task_SA_CA_VAE256_recon_64_First500ep' #_MA_001_Ori_001
# PATH_to_log_dir = './my_runs/Navigation/Single_Task/NewVer_500_episode/Old_3act_FloorPlan3_1Task_SA_CA_VAE256_recon_64'

# Device
CUDA_DEVICE_NUM = 0
DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
# print('Device:', DEVICE)


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

all_random_seed = 543
# set_all_seeds(all_random_seed)
# print('set random seed')


def tensor_to_RGB(tensor):
    image = tensor.cpu().clone()
    if image.dim()>3:
        image = image.squeeze(0)
    image = unloader(image)
    image = np.asarray(image).transpose(1,0,2)
    return image

# from VAE_NN import VAE
# from VAE256_siamese import VAE
from CBAM_VAE import VAE
# from CNN_Encoder_256 import CNN_Encoder

def CNN_Encoder_setup(Is_VAE, Weight_DIR, z_dim):
    if Is_VAE:
        # Load my VAE model weight
        model_Encoder = VAE(z_dim)
        model_Encoder.to(DEVICE)
        model_filename = 'VAE'                  # name for save model
        if DEVICE=='cpu':
            state_dict = torch.load(Weight_DIR, map_location=lambda storage, loc: storage)
        else:
            state_dict = torch.load(Weight_DIR, map_location=lambda storage, loc: storage.cuda(CUDA_DEVICE_NUM))

        model_Encoder.load_state_dict(state_dict)
        model_Encoder.eval()
        for p in model_Encoder.parameters():    # 固定VAE參數
            p.requires_grad = False
    # else:
    #     model_Encoder = CNN_Encoder()
    #     model_Encoder.to(DEVICE)
    #     model_filename = 'Encoder'
    #     model_Encoder.train()
    #     model_Encoder.share_memory()

    print(model_Encoder)
    return model_Encoder, model_filename

def make_env():
    def _thunk():
        env = ActiveVisionDatasetEnv()
        return env
    return _thunk


# A3C global NN
class Global_Net(nn.Module):
    def __init__(self, a_dim):
        super(Global_Net, self).__init__()
        # self.s_dim = s_dim
        self.a_dim = a_dim
        self.FC = nn.Sequential(
            # nn.Linear(512*3, 512),      # 512
            nn.Linear(64*3, 64),      # 64
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.Linear(512, 512),        #512
            nn.Linear(64, 64),        #64
            nn.ReLU()
        )

        # self.fpre_a = nn.Linear(3, 512) # action 3 -> 512
        self.fpre_a = nn.Linear(5, 64) # action 3 -> 64
        # self.fpre_a = nn.Linear(5, 64)  # action 5 -> 64    TEST action 3->5

        # self.actor = nn.Linear(512, a_dim)              # output action 3 [0,...1]
        # self.critic = nn.Linear(512, 1)                 # output value 1
        self.actor = nn.Linear(64, a_dim)              # output action 3 [0,...1]
        self.critic = nn.Linear(64, 1)                 # output value 1
        set_init([self.FC[0], self.FC[2], self.fpre_a, self.actor, self.critic])
        self.distribution = torch.distributions.Categorical

    def forward(self, state_fmap, pre_act):
        act_fmap = F.relu(self.fpre_a(pre_act))         # 3->64
        state_fmap = torch.cat((state_fmap, act_fmap), 1)                # 64*2 + 64 ->64*3
        fc_state_fmap = torch.tanh(self.FC(state_fmap))                 # 64*3 ->64

        # Actor
        logits = self.actor(fc_state_fmap)              # 64 -> 3
        # Critic
        values = self.critic(fc_state_fmap)             # 64 -> 1
        return logits, values

    def choose_act(self, state_fmap, pre_act, last_collide):       # state_fmap:64*2     2022/07/19 add last_collide ##
        self.eval()
        logits, _ = self.forward(state_fmap, pre_act)

        ### Modify  if last action collide => don't choose action:2(MoveAhead) ###
        if logits.shape[1] == 3:     ### Action space : 3 ###
            if last_collide == True:
                logits = logits[:, 0:2]
                probs = F.softmax(logits, dim=1)
                action = probs.multinomial(1).view(-1)[0].data
            else:
                probs = F.softmax(logits, dim=1)
                action = probs.multinomial(1).view(-1)[0].data
        else:                  ### Action space : 5 ###
            if last_collide == True:
                ar2 = torch.flatten(logits[:, 3:5])
                ar1 = torch.zeros(3).to(DEVICE)
                logits = torch.cat((ar2, ar1), dim=0)
                logits = torch.reshape(logits, (1, 5))
                probs = F.softmax(logits, dim=1)
                action = probs.multinomial(1).view(-1)[0].data
            else:
                probs = F.softmax(logits, dim=1)
                action = probs.multinomial(1).view(-1)[0].data
        ####################################################################################

        # probs = F.softmax(logits, dim=1)
        # action = probs.multinomial(1).view(-1)[0].data
        # # print('Policy Probs: ', probs.data, ' Action: ', action.item())
        return action.cpu().numpy().item()

    def loss_func(self, s, s_a, a, v_t):
        self.train()
        logits, values = self.forward(s, s_a)                # state: cur + target fmap ,此結果與執行step時會相同(因為weight參數沒變又eval)

        # critic loss : TD error (rt +gamma*Vt+1 -Vt)  (^2)
        td = v_t - values
        c_loss = td.pow(2)

        # actor loss : -TD * log(pi(st,at))
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v

        total_loss = (c_loss + a_loss).mean()
        return total_loss

# A3C local NN
class Worker(mp.Process):
    def __init__(self, N_S, N_A, MAX_Global_Ep, Set_resize, L_5, MAX_task_step, Self_Adjust_task_step_N, UPDATE_GLOBAL_ITER, GAMMA, gnet, opt,
                 model_Encoder, global_ep, global_ep_r, res_queue, global_all_act, lock, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        # self.agent_name = name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        # self.local_net = Global_Net(N_S,N_A)           # global to local network
        self.local_net = Global_Net(N_A)                # global to local network
        self.local_net.to(DEVICE)
        self.MAX_Global_Ep = MAX_Global_Ep
        self.Self_Adjust_N = Self_Adjust_task_step_N
        self.Set_resize = Set_resize
        self.L_5 = L_5
        self.MAX_task_step = MAX_task_step
        self.UPDATE_GLOBAL_ITER = UPDATE_GLOBAL_ITER
        self.GAMMA = GAMMA
        self.global_all_act, self.lock = global_all_act, lock
        self.model_Encoder = model_Encoder
        # if self.agent_name % 4==0: self._cur_world = random.choice(Env_FloorPlan[:3])
        # elif self.agent_name % 4==1: self._cur_world = random.choice(Env_FloorPlan[3:6])
        # elif self.agent_name % 4==2: self._cur_world = random.choice(Env_FloorPlan[6:9])
        # elif self.agent_name % 4==3: self._cur_world = random.choice(Env_FloorPlan[9:])
        # print(self._cur_world)
        self.target_record = []

    def run(self):
        self.env = ActiveVisionDatasetEnv()

        # k = 3                      # kitchen:1~30, Living roon:201~230, Bedroom:301~330, Bathroom:401~430
        # k = 206
        k = 303
        # k = 410                     # scene (environment) setting
        _cur_world = 'FloorPlan' + str(k)                       # 決定sence
        # cur_obs, target_obs, shortest_len, pre_action, _ = self.env.reset(_cur_world, self.MAX_task_step, self.Self_Adjust_N,  self.L_5)
        # cur_obs = torch.Tensor(cv2.resize(cur_obs.numpy(), self.Set_resize)).transpose(0, 2)    # [300,300,3] -> [3,128,128]
        # target_obs = torch.Tensor(cv2.resize(target_obs.numpy(), self.Set_resize)).transpose(0, 2)
        # self.cur_obs_ = cur_obs[None, :].to(DEVICE)             # [3,128,128] -> [1,3,128,128]
        # self.target_obs_ = target_obs[None, :].to(DEVICE)
        # self.pre_action_ = pre_action[None, :].to(DEVICE)       # 6 -> [1, 6]
        self.flag = False
        while self.g_ep.value < self.MAX_Global_Ep:
            Ep_step = 1
            buffer_s, buffer_a, buffer_r = [], [], []
            buffer_sa = []
            ep_r = 0.
            collide_ep = 0
            local_ep_act = []

            cur_obs, target_obs, shortest_len, pre_action, _, target_name = self.env.reset(_cur_world, self.MAX_task_step, self.Self_Adjust_N, self.L_5)

            # print(pre_action)
            # pre_action = torch.from_numpy(np.array([0, 0, 1], dtype=np.float32))

            # test_navi_start_time_point = time.time()  # only 測試運行速度與NN更新速度用  (到目前為止 當前time)

            self.target_record.append(target_name)
            #print(self.catarget_record)
            # plt.imshow(target_obs)
            # plt.show()
            cur_obs = torch.Tensor(cv2.resize(cur_obs.numpy(), self.Set_resize)).transpose(0, 2)  # [300,300,3] -> [3,128,128]
            target_obs = torch.Tensor(cv2.resize(target_obs.numpy(), self.Set_resize)).transpose(0, 2)
            self.cur_obs_ = cur_obs[None, :].to(DEVICE)  # [3,128,128] -> [1,3,128,128]
            self.target_obs_ = target_obs[None, :].to(DEVICE)
            self.pre_action_ = pre_action[None, :].to(DEVICE)  # 3 -> [1, 3]    * Env t size 3->5
            while True:
                if self.g_ep.value >= self.MAX_Global_Ep:       # 當所有Eposide結束,避免其他子程序在繼續執行
                    print('Shut Down subprocessing')
                    self.flag = True
                    break

                # input_state, recon_pre_obs = self.model_Encoder.Navigation_forward(self.cur_obs_, self.pre_action_, self.target_obs_)  #  VAE encode and Merge cur & target
                input_state = self.model_Encoder.Navigation_forward(self.cur_obs_, self.target_obs_)  #  VAE encode and Merge cur & target
                                                                                                                                        # if CNN_Encoder: recon_pre_obs = None
                if 'collided' not in locals():      # 2022/07/19 add last_collide(collided)
                    collided = False
                action = self.local_net.choose_act(input_state, self.pre_action_, collided)   # [0:RotateRight] [1:RotateLeft] [2:MoveAhead]
                # print('Worker: {}  |||  Colide: {} ||| Action: {}'.format(self.name, collided, ACTIONS[action]))
                # print(" Worker: {}   ---   Action = {}".format(self.name, ACTIONS[action]))

                new_obs, collided, r, done, succeed = self.env.step(idx=action, cur_ep=self.g_ep.value, max_ep=self.MAX_Global_Ep)   # 從環境中 get state, reward ,done or not

                # print('done:', done)
                # if done: r = -1
                ep_r += r
                local_ep_act.append(action)
                # if ep_r > 10:                                           # for debug
                #     print(ep_r)
                #     print('r:', r)
                #     print('=============ep_r Error=============')
                #     sys.exit()                                          # 強制停止

                buffer_a.append(action)
                buffer_s.append(input_state)                            # tensor
                buffer_r.append(r)
                # if self.name == 'w00':
                #     print('ep_r: ', round(ep_r, 3))
                #     print('action: ', action)
                    # writer.add_scalar('Reward/Episode', ep_r, self.g_ep.value)
                if collided:
                    collide_ep += 1

                act_onehot = torch.zeros(self.pre_action_.shape)        # shape [1, 3]
                cur_obs_ = torch.Tensor(cv2.resize(new_obs.numpy(), self.Set_resize)).transpose(0, 2)[None, :]
                act_onehot[0, action] = 1.

                self.cur_obs_ = cur_obs_.to(DEVICE)                     # [1,3,128,128]
                self.pre_action_ = act_onehot.to(DEVICE)                # [1, 3]
                buffer_sa.append(self.pre_action_)

                # update global and assign to local net (每UPDATE_GLOBAL_ITER次or done就更新一次golbal )
                if Ep_step % self.UPDATE_GLOBAL_ITER == 0 or done:
                    # if Ep_step==500:
                    #     Navi_finish_time_point = time.time()        # 到目前為止 當前time
                    #     print('Navigation '+str(Ep_step)+' step: %s sec' % (Navi_finish_time_point - test_navi_start_time_point))   # only for test Navigation speed for N step
                    #     print(local_ep_act)
                    #     print('for debug')

                    # sync
                    # s_, _ = self.model_Encoder.Navigation_forward(self.cur_obs_, self.target_obs_)
                    s_ = self.model_Encoder.Navigation_forward(self.cur_obs_, self.target_obs_)
                    pre_a = self.pre_action_
                    push_and_pull(self.opt, self.local_net, self.gnet, done, s_, pre_a, buffer_s, buffer_sa, buffer_a, buffer_r, self.GAMMA, DEVICE)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    buffer_sa = []
                    # if Ep_step==500:
                    #     UpdateNN_finish_time_point = time.time()    # 到目前為止 當前time
                    #     print('Update NN : %s sec' % (UpdateNN_finish_time_point - Navi_finish_time_point))   # only for test Navigation speed for N step
                    #     print('for debug')

                    if done:                                                        # done and print information
                        self.lock.acquire()                      # lock manager.list
                        with self.g_ep.get_lock():
                            with self.g_ep_r.get_lock():
                                print('----------------Done------------------')
                                ep_r = round(ep_r, 3)
                                if succeed:
                                    ep_spl = shortest_len/max(Ep_step, shortest_len)     # 計算單次Ep的SPL
                                else:
                                    ep_spl = 0
                                record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, collide_ep, succeed,
                                       self.global_all_act, local_ep_act, ep_spl, self.lock)
                                Ep_step += 1
                        self.lock.release()  # release manager lock
                        break

                Ep_step += 1
                # print('Ep_step :', Ep_step)

            # if not self.flag:
            #     # 每 N step or done(包含success) ,則reset Task
            #     rest_obs, rest_g_obs, shortest_len, rest_pre_act = self.env.reset_task()
            #     self.cur_obs_ = torch.Tensor(cv2.resize(rest_obs.numpy(), self.Set_resize)).transpose(0, 2)[None, :].to(DEVICE)
            #     self.target_obs_ = torch.Tensor(cv2.resize(rest_g_obs.numpy(), self.Set_resize)).transpose(0, 2)[None, :].to(DEVICE)
            #     self.act_onehot = rest_pre_act[None, :].to(DEVICE)
        # print('target_record: ', self.target_record)
        if not self.flag: self.res_queue.put([None, None, None, None, None, self.global_all_act, None])


if __name__ == '__main__':
# Env Setup
    Set_resize = (256, 256)                    # 原圖 300x300 -> Original image 300x300
    L_5 = True
    # MAX_task_step = 1000                       # 固定每個 Task 最大的 step 數
    MAX_task_step = None                       # 固定每個 Task 最大的 step 數 -> Fix the maximum number of steps per Task
    Self_Adjust_task_step_N = 20               # 自動調整每一個task的最大 step 數  (不是None則預設為True) -> Automatically adjust the maximum number of steps for each task (if not None, the default is True)

# a2c hyperparams:
    N_S = 1024                                 # obs shape (feature map pre, target)
    N_A = 5                                    # action space
    # N_A = 5                                    # action space
    UPDATE_GLOBAL_ITER = 10
    GAMMA = 0.9
    Per_Slice_Num = 100                         # 每 N Episode 計算平均 SR -> Calculate the average SR every N Episodes


    num_Worker = 2
    MAX_Global_Ep = 50                       # global Agent (total step 至少 MAX_task_step * Agent_Epoch次)
    # file_save_idx = 'Final_50.Fl410_all_rand'
    # file_save_idx = 'L5_Floor410_1Task_SA_CA_'
    file_save_idx = 'TEST_3act_Floor303_1Task_SA_CA_NewVer_'
    # file_save_idx = 'TEST.Modify_5act_Floor303_1Task_SA_CA_NewVer_Old_reward_First500ep_'

    # file_save_idx = 'Floor303_1Task_SA_CA_45D_R2G_1worker'
    VAE_z_dim = 64

# Model Setup
    Is_VAE = True

    # Weight_DIR = './VAE_recon_weight/24.FloorPlan3_VAE_recon_64z_1-256_batch256.pt'               # FloorPlan3 recon xt
    # Weight_DIR = './VAE_recon_weight/35.VAE_recon_64z_4-256_batch256.pt'               # 4 scene recon xt
    # Weight_DIR = './VAE_recon_weight/35.VAE_recon_64z_4-256_batch256.pt'               # VAE 4 scene recon xt
    Weight_DIR = '6.SA_CA_VAE_recon_64z_4-256_batch256.pt'         # SA_CA_VAE 4scene recon xt (not sure if this is right weight)
    # Weight_DIR = './VAE_recon_weight/9.SA_CA_45D_VAE_recon_64z_4-256_batch256.pt'       # SA_CA_45D_VAE (45D dataset wright)

    model_Encoder, model_filename = CNN_Encoder_setup(Is_VAE, Weight_DIR, VAE_z_dim)  # VAE or traditional CNN Encoder
    print('CNN model: ', model_filename+str(Set_resize[0]))

    for name, param in model_Encoder.named_parameters():   # print出model_Encoder可/不可被更新的 NN layers weight (param.requires_grad=True or False)
        if param.requires_grad:
            print('%s training' % name)
        else:
            print('%s freeze ' % name)
    # print(model_Encoder.state_dict())                    # all model_Encoder weight

    gnet = Global_Net(N_A)                                 # global network
    gnet = gnet.to(DEVICE)
    gnet.share_memory()                                    # share the global parameters in multiprocessing
    
    for name, param in gnet.named_parameters():
        if param.requires_grad:
            print('%s training' % name)
        else:
            print('%s freeze ' % name)
            sys.exit(1)

# Model FileName for Save

    # A3C+Encoder
    if model_filename == 'Encoder':
        if Self_Adjust_task_step_N is not None:
            Save_A3C_model_filename = 'A3C&' + model_filename + str(Set_resize[0])\
                                      + '-1_'+str(Self_Adjust_task_step_N)+'N_ep' \
                                      + str(MAX_Global_Ep) + '_.pt'
        else:
            Save_A3C_model_filename = 'A3C&' + model_filename + str(Set_resize[0]) + '-1_ep' + str(MAX_Global_Ep) + '_.pt'
        # Save_A3C_model_filename = 'FloorPlan3_A3C&' + model_filename + str(Set_resize[0]) + '-1_ep' + str(MAX_Global_Ep) + '_.pt'

    # A3C (based on VAE)
    else:
        if Self_Adjust_task_step_N is not None:
            Save_A3C_model_filename = file_save_idx + 'A3C_' + model_filename + str(Set_resize[0]) \
                                      + '-1_recon_'+str(Self_Adjust_task_step_N)+'N_ep' \
                                      + str(MAX_Global_Ep) + '_.pt'
        else:
            Save_A3C_model_filename = 'A3C_' + model_filename + str(Set_resize[0]) + '-1_ep' + str(MAX_Global_Ep) + '_.pt'
        # Save_A3C_model_filename = 'FloorPlan3_A3C_' + model_filename + str(Set_resize[0]) + '-1_ep' + str(MAX_Global_Ep) + '_.pt'


    # opt = SharedAdam(gnet.parameters(), lr=0.001, betas=(0.92, 0.999))                       # global optimizer
    opt = SharedAdam(gnet.parameters(), lr=7e-9, betas=(0.92, 0.999))                       # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()      # 使用Value將數據存在共享內存中 (i: 有符號整數 d: double浮點數) ,Queue存放子程式輸出值
    manager = Manager()
    global_all_act = manager.list([])
    lock = manager.Lock()


##### parallel training ##########

    workers = [Worker(N_S, N_A, MAX_Global_Ep, Set_resize, L_5, MAX_task_step, Self_Adjust_task_step_N, UPDATE_GLOBAL_ITER, GAMMA, gnet, opt,
                      model_Encoder, global_ep, global_ep_r, res_queue, global_all_act, lock, i) for i in range(num_Worker)]
    # [w.start() for w in workers]
    for w in workers:
        w.start()
        time.sleep(1)
    start_time = time.time()
    res = []  # record episode reward to plot
    ep_reward = []
    ep_collide_list = []
    succeed_list = []
    total_collide_cnt = 0
    Ep_spl = []
    Act_Count = Counter()  # 計數所有 ACTION(3) 個別執行次數
    total_step = 0

    # for show SR every N ep
    Per_Slice_Num = 100
    succeed_idx = 1
    succeed_reg_list = []
    writer = SummaryWriter(PATH_to_log_dir)
    while True:
        r, ep_r, collide_ep, succeed, ep_spl, global_all_act, act_counter = res_queue.get()
        if r is not None:  # [r, myr, None]
            res.append(r)
            ep_reward.append(ep_r)
            ep_collide_list.append(collide_ep)
            succeed_list.append(succeed)
            Ep_spl.append(ep_spl)
            Act_Count += act_counter  # 計算當前各Action數量
            total_step += sum(act_counter.values())  # 目前所有執行的step數 ,values()為取出各key(ACTION)的個數
            print('Collision/Step: ', collide_ep, '/', sum(act_counter.values()))
            print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

            writer.add_scalar('Reward / Episode', ep_r, global_ep.value)
            writer.add_scalar('Ep_SPL / Episode', ep_spl, global_ep.value)

            # for show SR every N ep
            succeed_reg_list.append(succeed)
            if len(succeed_reg_list) % Per_Slice_Num == 0:
                print(len(succeed_reg_list))
                s_success_rate = succeed_reg_list.count(True) * 100 / len(succeed_reg_list)
                # writer.add_scalar('Average success rate/Per_Slice_Num', s_success_rate, succeed_idx)
                succeed_idx += 1
                succeed_reg_list = []
        else:
            break

    [w.join() for w in workers]  # join 為阻塞當前的程序,直到呼叫join的那個子程序執行完,再繼續執行當前程序

    # Save Model
    # A3C+Encoder
    if model_filename == 'Encoder':
        torch.save({
            'gnet.state_dict': gnet.state_dict(),
            model_filename + str(Set_resize[0]) + '.state_dict': model_Encoder.state_dict()},
            Save_A3C_model_filename)

    # A3C (base on VAE)
    else:
        torch.save(gnet.state_dict(), Save_A3C_model_filename)

    # Analysis and Statistics
    total_collide_cnt = sum(ep_collide_list)  # total Collision
    collide_rate = total_collide_cnt * 100 / total_step  # Collision Rate
    succeed_num = succeed_list.count(True)  # SR
    succeed_rate = succeed_num * 100 / len(succeed_list)
    SPL = sum(Ep_spl) / len(Ep_spl)  # final SPL

    # print('res length', len(res))
    # print('res: ', res)
    # plt.plot(res)
    # plt.ylabel('Moving average ep reward')
    # plt.xlabel('Step')
    # plt.show(block=False)

    plt.plot(ep_reward)
    plt.ylabel('global every episode reward')
    plt.xlabel('global episodes')
    plt.savefig('myplot.png')
    plt.show(block=False)

    # Per_Slice_Num = 10
    s_success = []
    slice = [succeed_list[i:i + Per_Slice_Num] for i in range(0, len(succeed_list), Per_Slice_Num)]  # Slice SR
    for i in range(len(slice)):
        s_success_rate = slice[i].count(True) * 100 / len(slice[i])
        s_success.append(s_success_rate)
        writer.add_scalar('Average success rate/Per_Slice_Num', s_success_rate, i)
    print('Slice length: ', len(s_success))
    print(s_success)
    # plt.plot(s_success)
    # plt.ylabel('Average success rate')
    # plt.xlabel('Every ' + str(Per_Slice_Num) + ' global episodes')13.

    print('all_act: ', global_all_act[-2:])  # Episode action(Max N step)
    print('最後一筆Ep_act: ', len(global_all_act[-1]))
    print('Action total: ', sorted(Act_Count.items()))  # show all number of actions
    print('Global all_act Length: ', len(global_all_act))
    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    print('Collision_ep: ', ep_collide_list)

    print('ep_reward', ep_reward)
    print('ep_reward length ', len(ep_reward))
    print('Total step: ', total_step)
    print('Total Num of collision :', total_collide_cnt)
    print('Collision Rate: %.2f%%' % collide_rate)
    print('Success Rate: %.2f%%' % succeed_rate)
    print('Slice min/max SR: ', min(s_success), max(s_success))
    print('SPL: ', SPL)
    print('Using VAE: ', Is_VAE)
    print('Episodes: ', MAX_Global_Ep)
    print('num_Worker: ', num_Worker)
    print('Save Model FileName: ', Save_A3C_model_filename)
    print('for Debug')

    plt.show(block=True)





