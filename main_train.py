#### IMPORTS AREA ####

import warnings, os, sys, random, torch, time, cv2, tqdm, gc

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.multiprocessing as mp

from collections import Counter
from shared_adam import SharedAdam2 as SharedAdam
from CSA_VAE import VAE
from env_setup import ActivateEnv
from torch.utils.tensorboard import SummaryWriter
from utils import push_and_pull, record, save_name_file, init_layer

#### IMPORTS AREA ####



##### TRYING TO SET DEVICES TO GPU ####
torch.cuda.empty_cache()
gc.collect()
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('##### WARNING: YOU ARE USING THE CPU! #####')
##### TRYING TO SET DEVICES TO GPU ####



#### INIT AREA ####
warnings.filterwarnings("ignore")
action_space = ['RotateBack', 'RotateRight90', 'RotateRight', 'MoveAhead', 'RotateLeft90', 'RotateLeft']
EP = 10000
PBAR = tqdm.tqdm(total = EP, desc='Current progress')
#### INIT AREA ####



#### SETTING ALL SEED TO CERTAIN VALUES SO WE REDUCE THE RANDOM EFFECT OF THE TRAINING ####
def set_all_seeds(seed=None):
    if seed != None:
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('All random seed had been set to -> {}.'.format(all_random_seed))
    else:
        print('All seed have been toally randomly initialized.')



#### SETUP THE VISUAL DECODER ####
def encoder_setup(vae, weigh_dir, z_dim, print_arch=False):
    if vae:
        encoder_net = VAE(z_dim)
        encoder_net.to(DEVICE)
        model_filename = 'VAE'                  
        if DEVICE=='cpu':
            state_dict = torch.load(weigh_dir, map_location=lambda storage, loc: storage)
        else:
            state_dict = torch.load(weigh_dir, map_location=lambda storage, loc: storage.cuda())

        encoder_net.load_state_dict(state_dict)
        encoder_net.eval()
        for p in encoder_net.parameters():    
            p.requires_grad = False
    else:
        encoder_net = CNN_Encoder()
        encoder_net.to(DEVICE)
        model_filename = 'Encoder'
        encoder_net.train()
        encoder_net.share_memory()

    if print_arch:
        print(encoder_net)

    return encoder_net, model_filename


# A3C global NN
class Global_Net(nn.Module):

    def __init__(self, action_dim, z_dim=64, upscale_dim=8, mid1=256, mid2=128, mid3=64, n_lstm=2, p=0):
        super(Global_Net, self).__init__()

        if upscale_dim>action_dim+1:
            self.upscale_layer = nn.Sequential(
                #nn.BatchNorm1d(action_dim+1),
                nn.Linear(action_dim+1, 2*upscale_dim),
                #nn.BatchNorm1d(2*upscale_dim),
                nn.ReLU6(),
                nn.Linear(2*upscale_dim, 2*upscale_dim),
                #nn.BatchNorm1d(2*upscale_dim),
                nn.ReLU6(),
                nn.Linear(2*upscale_dim, upscale_dim),
                #nn.BatchNorm1d(upscale_dim),
                nn.ReLU6(),
            )
            init_layer(self.upscale_layer) 
        else:
            upscale_dim = action_dim+1

        if n_lstm != 0:
            self.lstm = nn.LSTM(2*z_dim+upscale_dim, mid1, n_lstm, batch_first=True, dropout=p)
            self.before_lstm = nn.Sequential(
            nn.BatchNorm1d(2*z_dim+upscale_dim),
            nn.ReLU6()
            )
            init_layer(self.before_lstm) 
            self.after_lstm = nn.Sequential(
                nn.BatchNorm1d(mid1),
                nn.ReLU6()
            )
            init_layer(self.after_lstm) 
        else:
            mid1 = 2*z_dim + upscale_dim
        
        self.first_layer = nn.Sequential(
            nn.Linear(mid1, mid2),
            #nn.BatchNorm1d(mid2),
            nn.ReLU6(),
            #nn.Dropout(p=p),
            # nn.Linear(mid2, mid3),
            # nn.BatchNorm1d(mid3),
            # nn.ReLU6(),
            # nn.Linear(mid3, mid3*2),
            # nn.BatchNorm1d(mid3*2),
            # nn.ReLU6(),
            # nn.Linear(mid3*2, mid3),
            # nn.BatchNorm1d(mid3),
            # nn.ReLU6(),
            nn.Linear(mid2, mid3),
            #nn.BatchNorm1d(mid3),
            nn.ReLU6(),
            #nn.Dropout(p=p)
        )
        init_layer(self.first_layer)        

        self.actor = nn.Sequential(
            nn.Linear(mid3, action_dim),
            #nn.BatchNorm1d(action_dim)
        )
        init_layer(self.actor)                                     

        self.critic = nn.Sequential(
            nn.Linear(mid3, 1),
            #nn.BatchNorm1d(1)
        )
        init_layer(self.critic)                                      

        self.distribution = torch.distributions.Categorical
        self.action_dim = action_dim
        self.n_lstm = n_lstm
        self.upscale_dim = upscale_dim
        self.z_dim = z_dim
        self.mid1 = mid1
        self.mid2 = mid2
        self.mid3 = mid3
        self.entropy_coeff=0.01


    def forward(self, cur_z, target_z, pre_act, hx, cx):
        
        if self.upscale_dim > self.action_dim+1:
            act_map = self.upscale_layer(pre_act)                             
        else:
            act_map = pre_act

        x = torch.cat((cur_z, target_z), 1)
        x = torch.cat((x, act_map), 1)

        if self.n_lstm!=0:
            #x = self.before_lstm(x)
            x, (hx, cx)= self.lstm(x, (hx.detach(), cx.detach()))
            #x = self.after_lstm(x)
        else:
            hx, cx = 0, 0


        x = self.first_layer(x)                

        # Actor
        logits = self.actor(x)                               

        # Critic
        values = self.critic(x)                              

        return logits, values, (hx, cx)


    def choose_act(self, cur_z, target_z, pre_act, hx, cx):
        self.eval()
        logits, _, (hx, cx) = self.forward(cur_z, target_z, pre_act, hx, cx)
        
        #print(logits)

        probs = F.softmax(logits)

        #print(logits)

        action = probs.multinomial(1).view(-1)[0].data

        return action.item(), (hx, cx)


    def loss_func(self, cur_z, target_z, pre_a, hx, cx, action, v_t):
        
        self.train()
        
        logits, values, _ = self.forward(cur_z, target_z, pre_a, hx, cx)   

        probs = F.softmax(logits)

        td = v_t - values
       
        c_loss = td.pow(2)
        
        m = self.distribution(probs)
        
        action_loss = -(m.log_prob(action) * td.detach().squeeze())
        

        total_loss = (c_loss + action_loss).mean()
        
        return total_loss

# A3C local NN
class Worker(mp.Process):
    def __init__(self, action_dim, max_glob_ep, input_dim, length_limit, max_step, automax_step, backprop_iter, gamma, global_net, opt,
                 encoder_net, global_ep, res_queue, lock, name):
        super(Worker, self).__init__()
        self.name = 'W{}'.format(name)
        self.cur_world = 'FloorPlan303'
        
        self.g_ep, self.res_queue = global_ep, res_queue
        self.action_dim=action_dim
        
        self.global_net = global_net 
        self.global_net.to(DEVICE)
        self.local_net = Global_Net(action_dim=action_dim)
        self.local_net.to(DEVICE)                
        self.encoder_net = encoder_net
        self.encoder_net.to(DEVICE)
        self.clip_grad = 0.1
        self.optimizer = opt
        self.gamma = gamma
        
        self.max_glob_ep = max_glob_ep
        self.max_step = max_step
        self.automax_step = automax_step
        self.input_dim = input_dim
        self.length_limit = length_limit
        self.backprop_iter = backprop_iter
        self.lock = lock
        
        self.memory_cur_z, self.memory_target_z, self.memory_actions, self.memory_pre_actions, self.memory_rewards = [], [], [], [], []
        self.buffer_cur_z, self.buffer_target_z, self.buffer_actions, self.buffer_pre_actions, self.buffer_rewards = [], [], [], [], []
        self.target_values = []
        
        if self.local_net.n_lstm!=0:
            self.cx = torch.zeros(self.local_net.n_lstm, self.local_net.mid1, device=DEVICE, dtype=torch.float32)
            self.hx = torch.zeros(self.local_net.n_lstm, self.local_net.mid1, device=DEVICE, dtype=torch.float32)
        else:
            self.hx = 0
            self.cx = 0
        self.pre_action = torch.zeros(1, action_dim+1, device=DEVICE, dtype=torch.float32)

    def mem_clean(self):
        self.buffer_cur_z = self.memory_cur_z
        self.buffer_target_z = self.memory_target_z
        self.buffer_actions = self.memory_actions
        self.buffer_pre_actions = self.memory_pre_actions
        self.buffer_rewards = self.memory_rewards

        self.memory_cur_z, self.memory_target_z, self.memory_actions, self.memory_pre_actions, self.memory_rewards = [], [], [], [], []

    def mem_enchance(self):
        self.memory_cur_z = self.buffer_cur_z + self.memory_cur_z
        self.memory_target_z = self.buffer_target_z + self.memory_target_z
        self.memory_actions = self.buffer_actions + self.memory_actions
        self.memory_pre_actions = self.buffer_pre_actions + self.memory_pre_actions
        self.memory_rewards = self.buffer_rewards + self.memory_rewards

    def push_and_pull(self):
        value = 0

        # self.target_values = np.empty(shape=(0, 0))
        # for reward in self.memory_rewards[::-1]:
        #     value = reward + self.gamma * value
        #     self.target_values = np.append(self.target_values, value)
        # self.target_values = np.flip(self.target_values)

        value = 0

        self.target_values = []
        for reward in self.memory_rewards[::-1]:
            value = reward + self.gamma * value
            self.target_values.append(value)
        self.target_values.reverse()

        #self.target_values = (self.target_values-self.target_values.mean())/self.target_values.std()
        
        self.optimizer.zero_grad()

        loss = self.local_net.loss_func(
            torch.vstack(self.memory_cur_z),
            torch.vstack(self.memory_target_z),
            torch.vstack(self.memory_pre_actions),
            self.hx,
            self.cx,
            torch.tensor(self.memory_actions, device=DEVICE),
            torch.tensor(self.target_values, device=DEVICE)[:, None])

        loss.backward()
        for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
            gp._grad = lp.grad
        #nn_utils.clip_grad_norm_(self.local_net.parameters(), self.clip_grad)
        #nn_utils.clip_grad_norm_(self.global_net.parameters(), self.clip_grad)
        self.optimizer.step()

        self.local_net.load_state_dict(self.global_net.state_dict())

    def run(self):
        
        self.env = ActivateEnv(action_space, self.length_limit, self.max_step, self.automax_step)
        self.flag = False
        self.ep_spl = 0.
        self.episode_steps = 1
        self.ep_reward = 0.
        self.episode_collides = 0
        self.collided = False


        while self.g_ep.value < self.max_glob_ep:
            if self.g_ep.value % 100 == 0:
                torch.save(self.global_net.state_dict(), './temp_a3c_models/temp_a3c_model_E{}'.format(self.g_ep.value))
            self.episode_steps = 1
            self.ep_reward = 0.
            self.episode_collides = 0
            self.collided = False
            if self.local_net.n_lstm!=0:
                self.cx[:,:] = 0
                self.hx[:,:] = 0
            self.pre_action[:,:] = 0 


            self.cur_frame, self.target_frame, self.shortest_len, self.target_name = self.env.reset()
            self.target_obs = self.encoder_net.encoding_fn(self.target_frame)

            #self.cur_obs_  = torch.nn.functional.interpolate(cur_obs, size=self.input_dim).view(1,3,256,256).to(DEVICE) # [300,300,3] -> [3,128,128]
            #self.target_obs_ = torch.nn.functional.interpolate(target_obs, size=self.input_dim).view(1,3,256,256).to(DEVICE)
            
            while True:
                
                if self.g_ep.value >= self.max_glob_ep:
                    self.flag = True
                    break
                
                self.cur_obs = self.encoder_net.encoding_fn(self.cur_frame)

                self.action, (self.hx, self.cx) = self.local_net.choose_act(self.cur_obs, self.target_obs, self.pre_action, self.cx, self.hx)  
                self.cur_frame, self.collided, self.step_reward, self.done, self.succeed = self.env.step(idx=self.action)  
                self.ep_reward += self.step_reward
                self.pre_action[:, self.action_dim] = int(self.collided)
                self.episode_collides += int(self.collided)
                
                self.memory_actions.append(self.action)
                self.memory_pre_actions.append(self.pre_action)
                self.memory_cur_z.append(self.cur_obs)
                self.memory_target_z.append(self.target_obs)
                self.memory_rewards.append(self.step_reward)
                   
                #self.cur_obs_ = torch.nn.functional.interpolate(new_obs, size=self.input_dim).view(1,3,256,256).to(DEVICE)
                
                if self.episode_steps % self.backprop_iter == 0 or self.done:
                    # if len(self.memory_rewards)>0:
                    self.push_and_pull()
                    self.mem_clean()
                    # else:
                    #     self.mem_enchance()
                    #     self.push_and_pull()
                    #     self.mem_clean()
                    if self.done:
                        #self.mem_clean()
                        self.lock.acquire()
                        with self.g_ep.get_lock():
                            PBAR.n = self.g_ep.value
                            if self.succeed:
                                self.ep_spl = self.shortest_len/max(self.episode_steps*0.125, self.shortest_len)
                            else:
                                self.ep_spl = 0
                            self.res_queue.put([self.succeed, self.ep_reward, self.ep_spl, self.episode_collides])
                            tqdm.tqdm.write('E{} - {} - {} | Succ:{} | Coll:{} | SPL:{:.2f} | EpR:{:.2f} | Target:{}'.format(
                                self.g_ep.value,
                                self.name, 
                                self.env.max_step,
                                self.succeed, 
                                self.episode_collides, 
                                self.ep_spl, 
                                self.ep_reward,
                                self.env.goal_name))
                            self.g_ep.value += 1
                            torch.cuda.empty_cache()
                            gc.collect()
                        self.lock.release()
                        break
                self.pre_action[:, :] = 0
                self.pre_action[:, self.action] = 1
                self.episode_steps += 1
        if not self.flag: self.res_queue.put([None, None, None, None])


if __name__ == '__main__':

#### Enviorment Setup ####
    action_dim = len(action_space) 
    all_random_seed = None                          # Seeds value
    set_all_seeds(seed=all_random_seed)             # Sets all seeds to certain value
    input_dim = (3, 256, 256)                       # Original image 300x300 -> resize to 256x256
    length_limit = True                            # Not to spawn too close to the target
    max_step = 500                                   # Fix the maximum number of steps per Task
    automax_step = None                             # Automatically adjust the maximum number of steps for each task (if not None, the default is True)


#### A2C Hyperparams Setup ####
    backprop_iter = 10                                    # Update training stats every X steps
    gamma = 0.9                                           # Discount factor for the rewards, range=(0, 1)
    Per_Slice_Num = 10                                    # Calculate the average SR (possibly succes rate) every N Episodes
    num_Worker = 4                                        # Number of ongoing simulations and A2C agents at the same time
    max_glob_ep = EP                                      # Global Agent (total step max_step * Agent_Epoch)
    file_save_idx = '{}-Act_Floor303'.format(action_dim)
    vae_z_dim = 64                                        # VAE sampled latent vector dimensions  


#### CNN Decoder Model Setup ####
    vae = True                                                  # Used when importing the image decoder
    weigh_dir = './vae_models/VAE_B128_L1000.pt'            # SA_CA_VAE 4scene recon xt (not sure if this is right weight)

    encoder_net, model_filename = encoder_setup(vae, weigh_dir, vae_z_dim)  # VAE or traditional CNN Encoder
    print('CNN decoder loaded -> {}-{}'.format(model_filename, input_dim[1]))

    global_net = Global_Net(action_dim=action_dim).to(DEVICE)                              # Initialize the global network
    global_net.share_memory()                                            # Share the global parameters in multiprocessing
    
    for name, param in global_net.named_parameters():                    # Check for training availability
        if not param.requires_grad:
            print('#### FORCED EXIT ####')
            sys.exit(1)
    print('Global A3C net initialized sucessfully.')


#### Get Model FileName for Saving ####
    Save_A3C_model_filename = save_name_file(file_save_idx, model_filename, input_dim[1], max_glob_ep, automax_step)


#### Set the optimization algorithm for Shared ADAM ####
    opt = SharedAdam(global_net.parameters(), lr=1e-4)
    #  \/ Use Value to store data in shared memory (i: signed integer d: double floating point number), Queue stores the output value of the subroutine
    global_ep, res_queue = mp.Value('i', 0), mp.Queue()
    manager = mp.Manager()
    lock = manager.Lock()


#### Starrting Parallel training ####
    workers = [Worker(action_dim=action_dim, 
                      max_glob_ep=max_glob_ep, 
                      input_dim=input_dim, 
                      length_limit=length_limit, 
                      max_step=max_step, 
                      automax_step=automax_step, 
                      backprop_iter=backprop_iter, 
                      gamma=gamma, 
                      global_net=global_net, 
                      opt=opt,
                      encoder_net=encoder_net, 
                      global_ep=global_ep, 
                      res_queue=res_queue,
                      lock=lock, 
                      name=i+1
                      ) for i in range(num_Worker)]

    print('Starting {} workers:'.format(num_Worker))
    for w in workers:
        w.start()
        print('     {} started!'.format(w.name))
        time.sleep(1)

    start_time = time.time()
    reward_list = np.zeros(EP)
    succeed_list = np.zeros(EP, dtype=np.uint8)
    spl_list = np.zeros(EP)

    count = int(open("count.txt", "r").read())

    log_directory = './my_runs/Navigation/Single_Task/E{}_A{}_LSTM{}_C{}'.format(EP, len(action_space), workers[0].local_net.n_lstm, count)

    writer = SummaryWriter(log_directory)
    while True:
        succeed, ep_reward, ep_spl, coll = res_queue.get()
        if succeed is not None:
            reward_list[global_ep.value-1] = ep_reward
            spl_list[global_ep.value-1] = ep_spl
            succeed_list[global_ep.value-1] = int(succeed)
            writer.add_scalar('Succes Rate', int(succeed)*100, global_ep.value)
            writer.add_scalar('Reward', ep_reward, global_ep.value)
            writer.add_scalar('SPL', ep_spl, global_ep.value)
            writer.add_scalar('Collides', coll, global_ep.value)
        else:
            break

    [w.join() for w in workers]  # join ????????????????????????,????????????join???????????????????????????,???????????????????????????

    # Analysis and Statistics
    # total_collide_cnt = sum(ep_collide_list)  # total Collision
    # collide_rate = total_collide_cnt * 100 / total_step  # Collision Rate
    # succeed_num = succeed_list.count(True)  # SR
    # succeed_rate = succeed_num * 100 / len(succeed_list)
    # SPL = sum(Ep_spl) / len(Ep_spl)  # final SPL

    # plt.plot(ep_rewards)
    # plt.ylabel('global every episode reward')
    # plt.xlabel('global episodes')
    # plt.savefig('myplot.png')
    # plt.show(block=False)

    # s_success = []
    # slice = [succeed_list[i:i + Per_Slice_Num] for i in range(0, len(succeed_list), Per_Slice_Num)]  # Slice SR
    # for i in range(len(slice)):
    #     s_success_rate = slice[i].count(True) * 100 / len(slice[i])
    #     s_success.append(s_success_rate)
    #     writer.add_scalar('Average success rate/Per_Slice_Num', s_success_rate, i)
    # print('Slice length: ', len(s_success))
    # print(s_success)

    # print('Action total: ', sorted(Act_Count.items()))  # show all number of action_space
    # print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    # # print('Collision_ep: ', ep_collide_list)

    succes_rate = np.sum(succeed_list)/EP
    print('Average SPL: {:.2f}'.format(np.average(spl_list)))
    print('Average reward: {:.2f}'.format(np.average(reward_list)))
    print('Success rate: {:.0f}%'.format(succes_rate*100))

    Save_A3C_model_filename = 'a3c_models/'+Save_A3C_model_filename+'sr{:.2f}_C{}.pt'.format(succes_rate*100, count)

    # print('ep_reward', ep_rewards)
    # print('ep_reward length ', len(ep_rewards))
    # print('Total step: ', total_step)
    # print('Total Num of collision :', total_collide_cnt)
    # print('Collision Rate: %.2f%%' % collide_rate)
    # print('Success Rate: %.2f%%' % succeed_rate)
    # print('Slice min/max SR: ', min(s_success), max(s_success))
    # print('SPL: ', SPL)
    # print('Episodes: ', max_glob_ep)
    # print('num_Worker: ', num_Worker)
    # print('Save Model FileName: ', Save_A3C_model_filename)



        # Save Model
    # A3C+Encoder
    if model_filename == 'Encoder':
        torch.save({
            'global_net.state_dict': global_net.state_dict(),
            model_filename + str(input_dim[1]) + '.state_dict': encoder_net.state_dict()},
            Save_A3C_model_filename)

    # A3C (base on VAE)
    else:
        torch.save(global_net.state_dict(), Save_A3C_model_filename)

    print('Saved as: {}'.format(Save_A3C_model_filename))

    count += 1

    open("count.txt", "w").write(str(count))

    # plt.show(block=True)





