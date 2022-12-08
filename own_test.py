#### IMPORTS AREA ####

import warnings
warnings.filterwarnings("ignore")

import os, sys, random, torch, time, cv2

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from collections import Counter
from shared_adam import SharedAdam
from CBAM_VAE import VAE
from Ai2thor_Env_Setup import ActiveVisionDatasetEnv
from torch.multiprocessing import Pool, Process, Manager, set_start_method
from torch.utils.tensorboard import SummaryWriter
from utils import tensor_to_RGB, v_wrap, set_init, push_and_pull, record, save_name_file

#### IMPORTS AREA ####



##### TRYING TO SET DEVICES TO GPU ####
CUDA_DEVICE_NUM = 0
try:
     set_start_method('spawn')
except RuntimeError:
    pass
DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')

if not torch.cuda.is_available():
    print('##### WARNING: YOU ARE USING THE CPU! #####')



#### SETTING ALL SEED TO CERTAIN VALUES SO WE REDUCE THE RANDOM EFFECT OF THE TRAINING ####
def set_all_seeds(seed=0):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('All random seed had been set to -> {}.'.format(all_random_seed))



#### SETUP THE VISUAL DECODER ####
def CNN_Encoder_setup(Is_VAE, Weight_DIR, z_dim, print_arch=False):
    if Is_VAE:
        # Load my VAE model weight
        model_Encoder = VAE(z_dim)
        model_Encoder.to(DEVICE)
        model_filename = 'VAE'                  # Name for save model
        if DEVICE=='cpu':
            state_dict = torch.load(Weight_DIR, map_location=lambda storage, loc: storage)
        else:
            state_dict = torch.load(Weight_DIR, map_location=lambda storage, loc: storage.cuda(CUDA_DEVICE_NUM))

        model_Encoder.load_state_dict(state_dict)
        model_Encoder.eval()
        for p in model_Encoder.parameters():    # Fixed VAE parameters
            p.requires_grad = False
    else:
        model_Encoder = CNN_Encoder()
        model_Encoder.to(DEVICE)
        model_filename = 'Encoder'
        model_Encoder.train()
        model_Encoder.share_memory()

    if print_arch:
        print(model_Encoder)

    return model_Encoder, model_filename



#### PROBABLY CREATE THE ENVIORMENT IN AI2THOR ####
def make_env():
    def _thunk():
        env = ActiveVisionDatasetEnv()
        return env
    return _thunk



#### A3C global Neural Network --- Here I start to add some modularity to the Class ####
class Global_Net(nn.Module):
    def __init__(self, n_actions, input_dims, gamma=0.99):
        super(Global_Net, self).__init__()
        self.gamma = gamma
        self.pi1 = nn.Linear(*input_dims, 64)                  # 1 Policy Layers, used to learn the behavior of the actor
        self.v1 = nn.Linear(*input_dims, 64)                   # 1 Value Layers, used to learn the behavior of the critic

        self.pi = nn.Linear(64, n_actions)                     # 2 Policy Layers, used to learn the behavior of the actor
        self.v = nn.Linear(64, 1)                              # 2 Value Layers, used to learn the behavior of the critic

        self.reward = []                                       # Used as memory buffers, easy to implement as simple lists
        self.actions = []                                      
        self.states = []


    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []


    def forward(self, state):

        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v


    def calc_R(self, done):
        states = torch.tensor(self.states, dtype=torch.float)
        _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        return batch_return


    def calc_loss(self, done):
        self.train()
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss


    def choose_act(self, state, last_collide):       # state_fmap:64*2     2022/07/19 add last_collide ##
        logits, _ = self.forward(state)

        ### Modify  if last action collide => don't choose action:2(MoveAhead) ###
        if logits.shape[1] == 3:                                    ### Action space : 3 ###
            if last_collide == True:
                logits = logits[:, 0:2]
                probs = F.softmax(logits, dim=1)
                action = probs.multinomial(1).view(-1)[0].data
            else:
                probs = F.softmax(logits, dim=1)
                action = probs.multinomial(1).view(-1)[0].data
        elif logits.shape[1] == 5:                                  ### Action space : 5 ###
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


#### A3C local Neural Network ####
class Worker(mp.Process):
    def __init__(self, N_S, N_A, MAX_Global_Ep, Set_resize, L_5, MAX_task_step, Self_Adjust_task_step_N, UPDATE_GLOBAL_ITER, GAMMA, gnet, opt,
                 model_Encoder, global_ep, global_ep_r, res_queue, global_all_act, lock, name):
        super(Worker, self).__init__()
        self.name = 'Worker-{}'.format(name)
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.local_net = Global_Net(n_actions=N_A, input_dims=[128])                # global to local network
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
        self.target_record = []


    def run(self):
        #### Create the enviroment ####
        self.env = ActiveVisionDatasetEnv()
        # \/ kitchen:1~30, Living roon:201~230, Bedroom:301~330, Bathroom:401~430
        k = 303                                                 # Enviorment settings                      
        _cur_world = 'FloorPlan' + str(k)                       # Decideed sence
        self.flag = False

        ### Episodes loop for MAX_Global_Ep values ####
        while self.g_ep.value < self.MAX_Global_Ep:
            Ep_step = 1
            self.local_net.clear_memory()
            ep_r = 0.
            collide_ep = 0
            local_ep_act = []

            cur_obs, target_obs, shortest_len, pre_action, _, target_name = self.env.reset(_cur_world, self.MAX_task_step, self.Self_Adjust_N, self.L_5)

            self.target_record.append(target_name)

            cur_obs = torch.Tensor(cv2.resize(cur_obs.numpy(), self.Set_resize)).transpose(0, 2)  # [300,300,3] -> [3,128,128]
            target_obs = torch.Tensor(cv2.resize(target_obs.numpy(), self.Set_resize)).transpose(0, 2)
            self.cur_obs_ = cur_obs[None, :].to(DEVICE)  # [3,128,128] -> [1,3,128,128]
            self.target_obs_ = target_obs[None, :].to(DEVICE)
            self.pre_action_ = pre_action[None, :].to(DEVICE)  # 3 -> [1, 3]    * Env t size 3->5
            if self.g_ep.value >= self.MAX_Global_Ep: 
                self.flag = True
                print('Shut Down subprocessing')
            while self.g_ep.value < self.MAX_Global_Ep:            # When all Eposides are over, avoid other subroutines from continuing to execute

                # input_state, recon_pre_obs = self.model_Encoder.Navigation_forward(self.cur_obs_, self.pre_action_, self.target_obs_)  #  VAE encode and Merge cur & target
                input_state = self.model_Encoder.Navigation_forward(self.cur_obs_, self.target_obs_)  #  VAE encode and Merge cur & target
                                                                                                                                        # if CNN_Encoder: recon_pre_obs = None
                if 'collided' not in locals():      # 2022/07/19 add last_collide(collided)
                    collided = False
                action = self.local_net.choose_act(input_state, collided)   # [0:RotateRight] [1:RotateLeft] [2:MoveAhead]
                # print("action = ", action)

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
                self.local_net.remember(state=input_state, action=action, reward=r)
                # if self.name == 'w00':
                #     print('ep_r: ', round(ep_r, 3))
                #     print('action: ', action)
                    # writer.add_scalar('Reward/Episode', ep_r, self.g_ep.value)
                if collided:
                    collide_ep += 1

                cur_obs_ = torch.Tensor(cv2.resize(new_obs.numpy(), self.Set_resize)).transpose(0, 2)[None, :]
                self.cur_obs_ = cur_obs_.to(DEVICE)                     # [1,3,128,128]

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
                    # push_and_pull(self.opt, self.local_net, self.gnet, done, s_, pre_a, buffer_s, buffer_sa, buffer_a, buffer_r, self.GAMMA, DEVICE)
                    # buffer_s, buffer_a, buffer_r = [], [], []
                    # buffer_sa = []
                    # if Ep_step==50push_and_pull0:
                    #     UpdateNN_finish_time_point = time.time()    # 到目前為止 當前time
                    #     print('Update NN : %s sec' % (UpdateNN_finish_time_point - Navi_finish_time_point))   # only for test Navigation speed for N step
                    #     print('for debug')

                    loss = self.local_net.calc_loss(done)
                    self.opt.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_net.parameters(),
                            self.gnet.parameters()):
                        global_param._grad = local_param.grad
                    self.opt.step()
                    self.local_net.load_state_dict(
                            self.gnet.state_dict())
                    self.local_net.clear_memory()

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
        print('target_record: ', self.target_record)
        if not self.flag: self.res_queue.put([None, None, None, None, None, self.global_all_act, None])





if __name__ == '__main__':


#### Enviorment Setup ####
    all_random_seed = 543                      # Seeds value
    set_all_seeds(seed=all_random_seed)        # Sets all seeds to certain value
    Set_resize = (256, 256)                    # Original image 300x300 -> resize to 256x256
    L_5 = False                                # Still now clue what this does
    # MAX_task_step = 1000                     # Fix the maximum number of steps per Task
    MAX_task_step = None                       # Fix the maximum number of steps per Task
    Self_Adjust_task_step_N = 50               # Automatically adjust the maximum number of steps for each task (if not None, the default is True)
    action_space = ['RotateRight', 'RotateLeft', 'MoveAhead', 'RotateRight90', 'RotateLeft90']



#### A2C Hyperparams Setup ####
    N_S = 1024                                 # Observations Shape (feature map pre, target)
    N_A = len(action_space)                    # Action Space (namely 3 actions -> Forward, Left, Right)
    # N_A = 5                                  # Action Space (namely 5 actions -> Forward, Left45, Left, Right45, Right)
    UPDATE_GLOBAL_ITER = 10                    # Update training stats every X steps
    GAMMA = 0.9                                # Discount factor for the rewards, range=(0, 1)
    Per_Slice_Num = 10                         # Calculate the average SR (possibly succes rate) every N Episodes
    num_Worker = 2                             # Number of ongoing simulations and A2C agents at the same time
    MAX_Global_Ep = 1000                       # Global Agent (total step MAX_task_step * Agent_Epoch)
    file_save_idx = 'TEST_3act_Floor303_1Task_SA_CA_NewVer_'
    VAE_z_dim = 64                             # VAE sampled latent vector dimensions                   



#### CNN Decoder Model Setup ####
    Is_VAE = True                                                  # Used when importing the image decoder
    Weight_DIR = '6.SA_CA_VAE_recon_64z_4-256_batch256.pt'         # SA_CA_VAE 4scene recon xt (not sure if this is right weight)

    model_Encoder, model_filename = CNN_Encoder_setup(Is_VAE, Weight_DIR, VAE_z_dim)  # VAE or traditional CNN Encoder
    print('CNN decoder loaded -> {}'.format(model_filename+'-'+str(Set_resize[0])))

    gnet = Global_Net(n_actions=N_A, input_dims=[128]).to(DEVICE)                              # Initialize the global network
    gnet.share_memory()                                            # Share the global parameters in multiprocessing
    
    for name, param in gnet.named_parameters():                    # Check for training availability
        if not param.requires_grad:
            print('#### FORCED EXIT ####')
            sys.exit(1)



#### Get Model FileName for Saving ####
    Save_A3C_model_filename = save_name_file(file_save_idx, model_filename, Set_resize[0], MAX_Global_Ep, Self_Adjust_task_step_N)
    


#### Set the optimization algorithm for Shared ADAM ####
    opt = SharedAdam(gnet.parameters(), lr=7e-4, betas=(0.92, 0.999))                        # global optimizer
 #  \/ Use Value to store data in shared memory (i: signed integer d: double floating point number), Queue stores the output value of the subroutine
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()      
    manager = Manager()
    global_all_act = manager.list([])
    lock = manager.Lock()



#### Starrting Parallel training ####
    workers = [Worker(N_S=N_S, 
                      N_A=N_A, 
                      MAX_Global_Ep=MAX_Global_Ep, 
                      Set_resize=Set_resize, 
                      L_5=L_5, 
                      MAX_task_step=MAX_task_step, 
                      Self_Adjust_task_step_N=Self_Adjust_task_step_N, 
                      UPDATE_GLOBAL_ITER=UPDATE_GLOBAL_ITER, 
                      GAMMA=GAMMA, 
                      gnet=gnet, 
                      opt=opt,
                      model_Encoder=model_Encoder, 
                      global_ep=global_ep, 
                      global_ep_r=global_ep_r, 
                      res_queue=res_queue, 
                      global_all_act=global_all_act, 
                      lock=lock, 
                      name=i
                      ) for i in range(num_Worker)]
    
    print('Starting {} workers:'.format(num_Worker))
    for w in workers:
        print('{} started!'.format(w.name))
        w.start()
        time.sleep(1)
    start_time = time.time()
    res = []                                                    # record episode reward to plot
    ep_reward = []
    ep_collide_list = []
    succeed_list = []
    total_collide_cnt = 0
    Ep_spl = []
    Act_Count = Counter()                                       # Count all ACTION(3) individual executions
    total_step = 0

# for show SR every N ep
    succeed_idx = 1
    succeed_reg_list = []
    PATH_to_log_dir = './my_runs/Navigation/Single_Task/NewVer_500_episode/Modify_5act_New_reward_FloorPlan303_1Task_SA_CA_VAE256_recon_64_First500ep'
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

    [w.join() for w in workers]  # join is to block the current program until the subprogram calling join is executed, and then continue to execute the current program

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