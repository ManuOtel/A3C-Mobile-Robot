import warnings
warnings.filterwarnings("ignore")
import os
from gym import spaces
import numpy as np
# from absl import logging440
import sys
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.animation as animation
import random

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
# import visualization_utils as vis_util

import ai2thor.controller
import torch

#TODO:
global i
i = 0



def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ActiveVisionDatasetEnv():
    """simulates the environment from ActiveVisionDataset."""
    cached_data = None

    def __init__(self, action_space, dataset_root='./aidata'):
        self.action_space = spaces.Discrete(len(action_space))  # len(ACTIONS)
        # self.observation_space = np.zeros([3, 300, 300])
        # self.observation_space = np.zeros([300, 300, 3])
        self._actions = action_space
        self.bc = ai2thor.controller.BFSController()
        self.bc.start()
        self.reset_num = 0
        # self.widx = 10
        # self.fidx = random.choice([0, 2, 3, 4, 2])
        self.record_act_R = 0       # record act for set reward
        self.record_act_L = 0
        self.record_act_MA = 0
        self.record_act_RL = 0
        self.record_act_HR = 0
        self.record_act_HL = 0
        self.D_r2g = 0
        self.first_ep = 1
        self._cur_world = None
        self.target_res = []
        self.test_disp_treja = False
        self.trej_x = []
        self.trej_y = []
        print('Enviroment succesfuly initiated!')

    def CheckAllGoal(self, _cur_world):
        self._cur_world = _cur_world
        print(self._cur_world)
        event = self.bc.reset(self._cur_world)
        self.bc.search_all_closed(self._cur_world)  # 找地圖所有網點並紀錄,當grid size固定 每次同一地圖找都是相同的點 (此grid size=0.25)
        self.graph = self.bc.build_graph()
        n_points = len(self.bc.grid_points)
        for idx in range(n_points):
            q = self.bc.key_for_point(self.bc.grid_points[idx])
            if q not in self.graph:
                print(self._cur_world, "false!!!")
                pause = input('Error, Agent not in graph. Press Enter to continue.')
                print('Continue')
        print("end!!")

    def target_setup_tool(self, _cur_world):
        while True:
            event = self.bc.reset(_cur_world)  # reset sence
            print('---Target Setup Tool----')
            if self._cur_world != _cur_world:
                self._cur_world = _cur_world
                self.bc.search_all_closed(self._cur_world)       # find grid points
                self.graph = self.bc.build_graph()          # grid points to graph
                self.goalkeys = self.bc.gkeys               # set my target from list
                self.tasktypenum = len(self.goalkeys)
                print(self.goalkeys)
            print('請輸入要設定之target index:')
            idx = int(input())
            target = self.goalkeys[idx]
            all = self.bc.goal[target]
            for i in range(len(all)):
                target_img = cv2.cvtColor(self.bc.goal[target][i]['frame'], cv2.COLOR_RGB2BGR)
                cv2.imshow(target + str(i), target_img)
            k = cv2.waitKey(0)
            if k == 27:  # 键盘上Esc键的键值
                cv2.destroyAllWindows()
            else:
                print('請輸入要儲存的 target image ID:')
                img_id = int(input())
                PATH_target_img = './target_img/'
                target_img = cv2.cvtColor(self.bc.goal[target][img_id]['frame'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(PATH_target_img+target+str(img_id)+'.jpg', target_img)
                print('img saved')
                cv2.destroyAllWindows()

    def Random_Set_target(self, goalkeys):
        if not self.target_res:                 # if self.target_res =[]
            self.target_res = list(goalkeys.items())
            random.shuffle(self.target_res)     # 直接執行 不用變數再存
        target = random.choice(self.target_res)
        self.target_res.remove(target)
        return target

    def reset(self, _cur_world, MAX_task_step=None, Self_Adjust_task_step_N=None, L_5=None):
        self.L_5 = L_5
        # print('--- Reset Env ---   L>=5 limit: ', self.L_5)
        self.step_count = 0
        event = self.bc.reset(_cur_world)  # reset sence
        # camera_event = self.bc.step(dict(action='ToggleMapView'))
        # cv2.imshow('Env Top View', camera_event.cv2img)
        # plt.imshow(camera_event.frame)
        # plt.title("event frame testing")
        # plt.show()
        if self._cur_world != _cur_world:
            self._cur_world = _cur_world
            # print('--- Setup grid points & Start to search all closed target and its position ---')
            self.bc.search_all_closed(self._cur_world)  # find grid points
            self.graph = self.bc.build_graph()  # grid points to graph
            self.goalkeys = self.bc.gkeys  # set my target from list
            self.tasktypenum = len(self.goalkeys)
            # print(self.goalkeys)
        self.reset_num += 1
        self.frame = 0
        self.reward = 0

        goal_info = self.Random_Set_target(self.goalkeys)
        # goal_info = list(self.goalkeys.items())[1]   # 1 task :Set Target Microwave

        self.goalname = goal_info[0]
        self.mygoal = self.bc.goal[self.goalname][goal_info[1]]
        self.mygoal['cameraHorizon'] = event.metadata['agent']['cameraHorizon']  # camera degree
        self.target_img = torch.Tensor(self.mygoal['frame'] / 255)#.transpose(0, 2)
        # print('target: ', self.goalname)
        n_points = len(self.bc.grid_points)

    # Task path length at least L setup
        while True:
            cidx = random.randint(0, n_points - 1)  # random設定起始位置與方向
            ri = random.randint(0, 3)
            # cidx = 11
        ## L = 13
        #     ri = 0
        ## L = 14
            # ri = 3

            # print('Set same start point cidx: ',cidx,' ri: ',ri )   # path len =14

            event = self.bc.step(
                dict(action='TeleportFull', x=self.bc.grid_points[cidx]['x'], y=self.bc.grid_points[cidx]['y'],
                     z=self.bc.grid_points[cidx]['z'], rotation=90.0 * ri, horizon=0.0))
            self.len_path = len(self.bc.shortest_plan(self.graph, event.metadata['agent'], self.mygoal))    # Move,Rotate 都算在內
            self.last_Geo = self.len_path           # 若有 Geo distance reward,作為一開始的Geo distance

        # Self Adjust MAX Step
            self.Self_Adjust_N = Self_Adjust_task_step_N
            self.MAX_task_step = MAX_task_step
            if self.Self_Adjust_N is not None:
                self.MAX_task_step = self.len_path * self.Self_Adjust_N                 # 依照倍率自動設定每一task MAX step 否則 依固定的task MAX step

        # L>=5 True
            if self.L_5:
                if (self.len_path >= 5) & (self.len_path < self.MAX_task_step):             # 設置正確 結束target設定迴圈
                    # print('---- L>=5 length limit ----  L=%s' % self.len_path, '  MAX Step: %s' % self.MAX_task_step)
                    break
                #elif self.len_path ==0:                                                 # 起點=終點 重設
                    # print(event.metadata['agent']['position'])
                    # print(self.mygoal['position'])
                    # print('==========Target-Start Point Setup Error, Reset Start Point========.')
                    # pause = input('or Press Enter to continue.')
                    # sys.exit()
                #else:
                    # print('L=%s ,Reset start point' % self.len_path,)                   # L太小 重設
        # No L limited
            else:
                #if self.len_path ==0:                                                   # 起點=終點 重設
                    # print(event.metadata['agent']['position'])
                    # print(self.mygoal['position'])
                    # print('==========Target-Start Point Setup Error, Reset Start Point========.')
                    # pause = input('or Press Enter to continue.')
                #else:
                    # print('---- No L length limit ----  L=%s' % self.len_path, '  MAX Step: %s' % self.MAX_task_step)   # 設置正確 結束target設定迴圈
                break
        self.cur_img = torch.Tensor(event.frame / 255)
        t = [0. for i in range(len(self._actions))]
        # t = [0., 0., 0., 0., 0.]
        self.pre_action = torch.from_numpy(np.array(t, dtype=np.float32))
        return self.cur_img, self.target_img, self.len_path, self.pre_action, self.bc.grid_points, self.goalname

    def step(self, idx, cur_ep, max_ep):                            # action is a digit
        # t = [0., 0., 0., 0., 0., 0.]
        t = np.zeros(len(self._actions))
        t[idx] = 1.
        # self.pre_action = torch.from_numpy(np.array(t, dtype=np.float32))
        self.reward = 0
        self.done = False
        self.collided = False
        self.succeed = False
        self.step_count += 1
        action = self._actions[idx]
        # action = self._actions[0]     # for debug
        if action == 'RotateRight' or action == 'RotateRight90':
            self.record_act_R += 1
            self.record_act_L = 0
            self.record_act_RL += 1
        elif action == 'RotateLeft' or action == 'RotateLeft90':
            self.record_act_R = 0
            self.record_act_L += 1
            self.record_act_RL += 1
        else:
            self.record_act_MA += 1
            self.record_act_R = 0
            self.record_act_L = 0
            self.record_act_RL = 0
        # print('R: %s' % self.record_act_R)
        # print('L: %s' % self.record_act_L)

        # Modify action
        if action == 'MoveAhead':
            event = self.bc.step(dict(action=action))  # Do action
        elif action == 'RotateRight':
            event = self.bc.step(action="RotateRight", degrees=45)  # Do action
        elif action == 'RotateLeft':
            event = self.bc.step(action="RotateLeft", degrees=45)  # Do action
        elif action == 'RotateRight90':
            event = self.bc.step(dict(action="RotateRight"))  # Do action
        elif action == 'RotateLeft90':
            event = self.bc.step(dict(action="RotateLeft"))  # Do action
        
        # event = self.bc.step(dict(action=action))    # Do action
        
        # test frame
        # plt.imshow(event.frame)
        # plt.title("event frame testing")
        # plt.show()

        # only for testing
        global i
        if self.test_disp_treja == True:
            self.trej_x.append(event.metadata['agent']['position']['x'])
            self.trej_y.append(event.metadata['agent']['position']['z'])
            plt.ion()
            if self.ax0.lines != []:
                # self.ax0.lines.remove(self.ax0.lines[0])  # 移除單一line
                self.ax0.lines.clear()                      # 移除所有lines
            self.ax0.plot(self.trej_x[-4:], self.trej_y[-4:], 'bo-',            # only show the last 4 traj
                          self.start_point_x, self.start_point_y, 'co',
                          self.mygoal['position']['x'], self.mygoal['position']['z'], 'ro',)
            # Show Current Orientation (As Arrow)
            rot = round(event.metadata['agent']['rotation']['y'], 0) / 45
            if rot == 0:
                self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], 0 * 0.25, 1 * 0.25, width=0.03, length_includes_head=True)
            elif rot == 1:
                self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], 1 * 0.17, 1 * 0.17, width=0.03, length_includes_head=True)
            elif rot == 2:
                self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], 1 * 0.25, 0 * 0.25, width=0.03, length_includes_head=True)
            elif rot == 3:
                self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], 1 * 0.17, -1 * 0.17, width=0.03, length_includes_head=True)
            elif rot == 4:
                self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], 0 * 0.25, -1 * 0.25, width=0.03, length_includes_head=True)
            elif rot == 5:
                self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], -1 * 0.17, -1 * 0.17, width=0.03, length_includes_head=True)
            elif rot == 6:
                self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], -1 * 0.25, 0 * 0.25, width=0.03, length_includes_head=True)
            elif rot == 7:
                self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], -1 * 0.17, 1 * 0.17, width=0.03, length_includes_head=True)
            else :
                print('Error: rot bug!')    # For debug
            # test save
            #TODO:
            self.fig_grid_diplay.savefig(os.path.dirname(__file__) +"/Testing/test_map/test_map" + str(i) + ".png")
            # print('event.metadeta (rotation) = ', round(event.metadata['agent']['rotation']['y'], 0))       # 軌跡圖方位(test_map)/45  0:上  1:右上  2:右 ... 7:左上
            plt.ioff()
            plt.pause(0.00001)

        #TODO:
        im1 = Image.fromarray(event.frame)
        im1.save(os.path.dirname(__file__)+"/Testing/test_frame/test_frame"+ str(i) + ".png","png")
        i += 1


        # cur_Geo = len(self.bc.shortest_plan(self.graph, event.metadata['agent'], self.mygoal))
        self.cimg = torch.Tensor(event.frame / 255)
        # self.prev_event = {'position': self.bc.last_event.metadata['agent']['position'],    # current state information (after action)
        #                    'rotation': self.bc.last_event.metadata['agent']['rotation'],
        #                    'cameraHorizon': self.bc.last_event.metadata['agent']['cameraHorizon']}

    # =======================================
        event = self.bc.last_event              # current state information (after action)


            #   到達target (用座標判斷)        
        if abs(event.metadata['agent']['position']['x'] - self.mygoal['position']['x']) <= 0.35 and \
           abs(event.metadata['agent']['position']['z'] - self.mygoal['position']['z']) <= 0.35 and \
           abs(event.metadata['agent']['rotation']['y'] - self.mygoal['rotation']['y']) == 0 :
            self.reward += 1000.0  ##  original = 10.0  ##
            # self.reward = 1.0     # 改成1.0測試結果是否有差
            self.succeed = True
            self.done = True
            # print('Succeed to get target')
            if self.test_disp_treja == True:
                plt.ion()
                self.ax0.plot(self.trej_x, self.trej_y, 'bo-')
                plt.ioff()
                plt.pause(0.001)

        if not self.succeed:
            self.reward += -1
        # 碰撞
            if not event.metadata['lastActionSuccess']:  # action是否成功執行,否則為判定碰撞
                self.collided = True
                # print('Collide')
                # self.reward += -0.01
                # self.reward += -0.2
                self.reward += -1
        # 原地打轉(連續某方向)
            # if self.record_act_R > 4:  # 原地連續右轉了>1圈 給懲罰 重置計數
            #     self.record_act_R = 0
            #     self.reward += -0.2
            #     # print('Over doing RotateRight')
            # if self.record_act_L > 4:  # 原地連續左轉了>1圈 給懲罰 重置計數
            #     self.record_act_L = 0
            #     self.reward += -0.2
                # print('Over doing RotateLeft')
            # if self.record_act_RL > 8:  # 原地左右轉累積超過8次 => 原地左右打轉 給懲罰 重置計數
            #     self.record_act_RL = 0
            #     self.reward += -0.2
            #     # print('Wiggle > 8')
            
        # N step 沒有前進
        #     N = 5
        #     if self.step_count % N == 0:  # N 步內若沒有前進 持續給懲罰
        #         if self.record_act_MA == 0:
        #             self.reward += -0.2
        #             print('No MoveAhead in ' + str(N) + ' steps')
        #         else:
        #             self.record_act_MA = 0
        # N step 有前進
            N = 5
            # if self.step_count % N == 0:
            #     if self.record_act_MA > 0:  # N 步內若有前進 給獎勵並重置計數
            #         self.reward += 0.2
            #         # print('MoveAhead in ' + str(N) + ' steps')

            #     self.record_act_MA = 0
        # 其他
        # # Reward : 機器人視野與"目標視野角度"一致( +45 ~ -45 give reward )
        #     if -45 < round(event.metadata['agent']['rotation']['y'] - self.mygoal['rotation']['y'], 0) < 45:
        #         self.reward += 0.01     # 0.1 -> 0.01
        #
        # # Reward : History shortest distance to goal
        #     self.D_r2g = ((event.metadata['agent']['position']['x'] - self.mygoal['position']['x'])**2 + (event.metadata['agent']['position']['z'] - self.mygoal['position']['z'])**2)**0.5
        #     # print("(測試)self.D_r2g = " + str(self.D_r2g))
        #     if self.first_ep == 1:   # first episode => D_shortest = D_r2g
        #         self.D_shortest = ((event.metadata['agent']['position']['x'] - self.mygoal['position']['x']) ** 2 + (
        #                 event.metadata['agent']['position']['z'] - self.mygoal['position']['z']) ** 2) ** 0.5
        #         self.first_ep = 0
        #     if self.D_r2g < self.D_shortest:
        #         # print("最短距離刷新 : " + str(self.D_r2g))
        #         # print("當前最短 - 歷史最短 : " + str(self.D_shortest - self.D_r2g))
        #         self.reward += ( self.D_shortest - self.D_r2g ) * 10     # 10->5
        #         self.D_shortest = self.D_r2g
            
        # Geo Distance
        #     self.reward += 0.5 * (self.last_Geo - cur_Geo)
        #     self.last_Geo = cur_Geo
        # Otherwise
        #     self.reward += -0.01
        #
            # others = np.round(-0.001 * (self.step_count / self.len_path), 3) # if L=14, r=-0.00714 ~ -10, if round:(-0.01 ~ -10.00)
            # n = 10
            # others = np.round(-0.001 * (self.step_count / n ), 3) # if L=14, r=-0.00714 ~ -10, if round:(-0.01 ~ -10.00)
            # self.reward += -0.01 + others
            # print(others)

        # print(self.reward)
        #===========================================
        p = self.bc.key_for_point(event.metadata['agent']['position'])
        if p not in self.graph:  # graph 為reset 得到的grid point建的,若點不在裡面則判定碰撞了
            # print("self.collided = ", self.collided)
            # print("============ p not in graph!!!==============")
            # self.reward = -10.0
            self.collided = True
            # self.reward += -0.01
            # print(self._cur_world, "false!!!")
            # print('Error, Agent not in graph. Press Enter to continue.')
            # sys.exit(1)

        if self.step_count == self.MAX_task_step:
            self.done = True
            if self.test_disp_treja == True:
                plt.ion()
                self.ax0.plot(self.trej_x, self.trej_y, 'ro-')
                # print('Fail')
                plt.ioff()
                plt.pause(0.001)

        return self.cimg, self.collided, self.reward, self.done, self.succeed

    def setup_testing_env(self, _cur_world):
        self.test_disp_treja = True
        self.testing_event = self.bc.reset(_cur_world)  # reset sence
        self._cur_world = _cur_world
        print('--- Setup grid points & Start to search all closed target and its position ---')
        self.bc.search_all_closed(self._cur_world)  # find grid points
        self.graph = self.bc.build_graph()  # grid points to graph
        self.goalkeys = self.bc.gkeys  # set my target from list

        # 1 Target
        # goal_info = list(self.goalkeys.items())[1]   # TEST :Set Target Microwave
        # self.goalname = goal_info[0]
        # all_goal_info = [goal_info]
        # -------
        # 5 Target
        print(self.goalkeys)
        all_goal_info = list(self.goalkeys.items())

        return all_goal_info

    def reset_for_testing(self, goal_info, MAX_task_step=None, Self_Adjust_task_step_N=None, L_5=None):
        self.trej_x = []
        self.trej_y = []
        points_list_x = []
        points_list_y = []
        print('Grid points: ', self.bc.grid_points)

        plt.ion()
        # self.fig_grid_diplay = plt.figure()
        self.fig_grid_diplay, self.ax0 = plt.subplots(1)
        # self.fig_grid_diplay.canvas.manager.window.wm_geometry('+1200+100')
        for point_info in self.bc.grid_points:
            # [p_x, p_y] = point_info['x'], point_info['z']
            p_x, p_y = point_info['x'], point_info['z']
            # points_list.append([p_x, p_y])
            points_list_x.append(p_x)
            points_list_y.append(p_y)
        print('all num of points :', len(points_list_x))
        # self.ax0 = self.fig_grid_diplay.add_subplot(1, 1, 1)
        x_major_locator = MultipleLocator(0.25)     # axes size 0.25x
        y_major_locator = MultipleLocator(0.25)
        self.ax0.xaxis.set_major_locator(x_major_locator)
        self.ax0.yaxis.set_major_locator(y_major_locator)
        self.ax0.scatter(points_list_x, points_list_y)
        # self.ax0.plot(points_list_x, points_list_y, 'o')
        plt.ioff()
        plt.pause(0.01)

        self.L_5 = L_5
        print('--- Reset Task for testing ---   L>=5 limit: ', self.L_5)
        self.step_count = 0
        self.goalname = goal_info[0]
        self.mygoal = self.bc.goal[self.goalname][goal_info[1]]     # 設定 target
        self.mygoal['cameraHorizon'] = self.testing_event.metadata['agent']['cameraHorizon']  # camera degree
        self.target_img = torch.Tensor(self.mygoal['frame'] / 255)  # .transpose(0, 2)
        print('target: ', self.goalname)
        targetImg_display = cv2.cvtColor(self.mygoal['frame'], cv2.COLOR_RGB2BGR)
        cv2.imshow('Target', targetImg_display)           # 測試時顯示target img
        cv2.waitKey(100)
        n_points = len(self.bc.grid_points)

        # 2021/12/15 test save target image
        #TODO:
        tar_im = Image.fromarray(self.mygoal['frame'])
        tar_im.save(os.path.dirname(__file__)+"/Testing/test_target/test_target"+ str(i) + ".png","png")

        # Task path length at least L setup
        while True:
            cidx = random.randint(0, n_points - 1)  # random設定起始位置與方向
            ri = random.randint(0, 3)
            #
            # cidx = 11
            # ri = 3
            print('Set same start point cidx: ',cidx,' ri: ',ri )   # path len =14

            event = self.bc.step(
                dict(action='TeleportFull', x=self.bc.grid_points[cidx]['x'], y=self.bc.grid_points[cidx]['y'],
                     z=self.bc.grid_points[cidx]['z'], rotation=90.0 * ri, horizon=0.0))
            self.start_point_x = self.bc.grid_points[cidx]['x']
            self.start_point_y = self.bc.grid_points[cidx]['z']
            self.len_path = len(
                self.bc.shortest_plan(self.graph, event.metadata['agent'], self.mygoal))  # Move,Rotate 都算在內

            # Self Adjust MAX Step
            self.Self_Adjust_N = Self_Adjust_task_step_N
            self.MAX_task_step = MAX_task_step
            if self.Self_Adjust_N is not None:
                self.MAX_task_step = self.len_path * self.Self_Adjust_N  # 依照倍率自動設定每一task MAX step 否則 依固定的task MAX step

            # L>=5 True
            if self.L_5:
                if (self.len_path >= 5) & (self.len_path < self.MAX_task_step):  # 設置正確 結束target設定迴圈
                    print('---- L>=5 length limit ----  L=%s' % self.len_path, '  MAX Step: %s' % self.MAX_task_step)
                    break
                elif self.len_path == 0:  # 起點=終點 重設
                    print(event.metadata['agent']['position'])
                    print(self.mygoal['position'])
                    print('==========Target-Start Point Setup Error, Reset Start Point========.')
                    # pause = input('or Press Enter to continue.')
                    # sys.exit()
                else:
                    print('L=%s ,Reset start point' % self.len_path, )  # L太小 重設
            # No L limited
            else:
                if self.len_path == 0:  # 起點=終點 重設
                    print(event.metadata['agent']['position'])
                    print(self.mygoal['position'])
                    print('==========Target-Start Point Setup Error, Reset Start Point========.')
                    # pause = input('or Press Enter to continue.')
                else:
                    print('---- No L length limit ----  L=%s' % self.len_path,
                          '  MAX Step: %s' % self.MAX_task_step)  # 設置正確 結束target設定迴圈
                    break
        self.cur_img = torch.Tensor(event.frame / 255)
        # t = [0., 0., 0., 0., 0., 0.]
        t = [0. for i in range(len(self._actions))]
        # t = [0., 0., 0., 0., 0.]    # N_A 3->5
        self.pre_action = torch.from_numpy(np.array(t, dtype=np.float32))
        return self.cur_img, self.target_img, self.len_path, self.pre_action, self.bc.grid_points, self.goalname


if __name__ == '__main__':
    env = ActiveVisionDatasetEnv()

    # my_scene=[1,3,4, 6,  202,205,206, 210,  302,303,304, 305,   406,407,410, 411]             # 6, 210, 305, 411 for testing
    k = 410
    _cur_world = 'FloorPlan' + str(k)
    print(_cur_world)
    env.target_setup_tool(_cur_world)


