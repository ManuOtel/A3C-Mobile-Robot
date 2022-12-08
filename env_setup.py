#### IMPORTS AREA ####
import warnings
warnings.filterwarnings("ignore")

import random, cv2, os, torch, ai2thor.controller, gc

import numpy as np
import matplotlib.pyplot as plt

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ai2thor.util.metrics import get_shortest_path_to_object_type, path_distance, vector_distance
from matplotlib.pyplot import MultipleLocator
from gym import spaces
from PIL import Image, ImageDraw, ImageFont
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
#### IMPORTS AREA ####

#TODO:
global i
i = 0



def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ActivateEnv():

    def __init__(self, action_space, length_limit=False, max_step=None, automax_step=None):
        self.action_space = action_space
        self.action_dim = len(action_space)
        self.cur_world = 'FloorPlan303'
        self.bc = Controller(scene=self.cur_world,
                             platform=CloudRendering,
                             snapToGrid=True,
                             width=300,
                             height=300,
                             fieldOfView=90)
        self.bc.step(dict(action='Initialize', gridSize=0.25))
        self.length_limit = length_limit
        self.max_step=max_step
        self.automax_step=automax_step
        self.reset_num = 0
        self.target_res = []
        self.test_disp_treja = False
        self.trej_x = []
        self.trej_y = []
        self.reward = 0
        self.done = False
        self.collided = False
        self.succeed = False
        self.step_count = 0
        self.ra = False
        self.event = None
        self.dist = 0
        self.last_distance = 0
        self.action = self.action_space[0]
        self.ma_rec = 0 
        self.rl_rec = 0
        self.goal_list = ['Desk']
        self.rotations = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        self.set_goal()
        print('Enviroment succesfuly initiated!')

    def CheckAllGoal(self, _cur_world):
        self._cur_world = _cur_world
        print(self._cur_world)
        event = self.bc.reset(self._cur_world)
        self.bc.search_all_closed(self._cur_world)
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
            event = self.bc.reset(_cur_world)  
            print('---Target Setup Tool----')
            if self._cur_world != _cur_world:
                self._cur_world = _cur_world
                self.bc.search_all_closed(self._cur_world)       # find grid points
                self.graph = self.bc.build_graph()          # grid points to graph
                self.goalkeys = self.bc.gkeys               # set my target from list
                self.tasktypenum = len(self.goalkeys)
                #print(self.goalkeys)
            #print('請輸入要設定之target index:')
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
                # print('請輸入要儲存的 target image ID:')
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
        #print(goalkeys)
        #print(target)
        return target

    def set_goal(self):
        self.goal_name = random.choice(self.goal_list)
        self.reach_pos = self.bc.step(dict(action='GetReachablePositions'))

        for obj2 in self.reach_pos.metadata["objects"]:
            if obj2["objectType"] == self.goal_name:
                self.goal = obj2

                tx = self.goal['position']['x']
                ty = self.goal['position']['y']
                tz = self.goal['position']['z']

                ry = round(self.goal['rotation']['y'])

                minn = float('inf')
                maxx = float('-inf')

                for i in self.reach_pos.metadata["actionReturn"]:
                    x = i['x']
                    z = i['z']
                    y = i['y'] 

                    if (abs(tx-x)+abs(tz-z)) < minn:
                        aux = x
                        auz = z
                        minn = (abs(tx-x)+abs(tz-z))
                    if (abs(tx-x)+abs(tz-z)) > maxx:
                        maux = x
                        mauz = z
                        maxx = (abs(tx-x)+abs(tz-z))

        ev = self.bc.step(
                    action="Teleport",
                    position=dict(x=aux,y=y, z=auz),
                    rotation=dict(x=0, y=ry, z=0),
                    horizon=0,
                    standing=True
                )
        self.bc.step(action="MoveAhead")
        ev = self.bc.step(action='RotateRight', degrees=180)
        self.goal['frame']=ev.frame
        self.goal['position']=ev.metadata['agent']['position']
        self.goal['rotation']=ev.metadata['agent']['rotation']

        self.big_distance = vector_distance(self.goal['position'], dict(x=maux, y=y, z=mauz))

        gc.collect()

    def reset(self):
        self.dist = 0
        self.last_distance = 0
        self.step_count = 0
        self.event = self.bc.reset()    

        self.reset_num += 1
        self.frame = 0
        self.reward = 0

        self.set_goal()

        #self.bc.start_search(scene=self._cur_world)

        while True:
            position = random.choice(self.reach_pos.metadata['actionReturn'])
            rotation = random.choice(self.rotations)
            self.bc.step(action="Teleport",
                          position=position,
                          rotation=dict(x=0, y=rotation, z=0),
                          horizon=0,
                          standing=True)

            event = self.bc.step(action="GetShortestPathToPoint",
                                      position=position,
                                      x=self.goal['position']["x"],
                                      y=self.goal['position']["y"],
                                      z=self.goal['position']["z"],)

            self.optimal_path = event.metadata["actionReturn"]['corners']
            self.optimal_path_length = path_distance(self.optimal_path)

            # Self Adjust MAX Step
            if self.automax_step is not None:
                self.max_step = int(self.optimal_path_length * self.automax_step)

            # L>=5 True
            if self.length_limit:
                if (self.optimal_path_length >= 1):
                    break
            else:
                break

        self.cur_img=torch.from_numpy(event.frame.copy()).float()
        self.target_img=torch.from_numpy(self.goal['frame'].copy()).float()
        #print(self.cur_img)
        gc.collect()
        return self.cur_img.view(1,3,300,300).cuda(), self.target_img.view(1,3,300,300).cuda(), self.optimal_path_length, self.goal_name

    def step(self, idx, save=False):
        global i                           
        self.reward = 0
        self.done = False
        self.collided = False
        self.succeed = False
        self.step_count += 1
        self.ma = True
        self.action = self.action_space[idx]
        match self.action:
            case 'MoveAhead':
                event = self.bc.step(action=self.action)
                self.ma = True
                self.ma_rec += 1 
                self.rl_rec = 0
            case 'RotateRight':
                event = self.bc.step(action="RotateRight", degrees=45)
                self.ma_rec = 0
                self.rl_rec += 1
            case 'RotateRight90':
                event = self.bc.step(action='RotateRight')
                self.ma_rec = 0
                self.rl_rec += 1
            case 'RotateLeft':
                event = self.bc.step(action="RotateLeft", degrees=45)
                self.ma_rec = 0
                self.rl_rec += 1
            case 'RotateLeft90':
                event = self.bc.step(action="RotateLeft")
                self.ma_rec = 0
                self.rl_rec += 1

        if self.test_disp_treja == True:
            self.trej_x.append(event.metadata['agent']['position']['x'])
            self.trej_y.append(event.metadata['agent']['position']['z'])
            plt.ion()
            if self.ax0.lines != []:
                self.ax0.lines.clear()
            self.ax0.plot(self.trej_x[-4:], self.trej_y[-4:], 'bo-', 
                          self.start_point_x, self.start_point_y, 'co',
                          self.goal['position']['x'], self.goal['position']['z'], 'ro',)
            # Show Current Orientation (As Arrow)
            rot = round(event.metadata['agent']['rotation']['y'], 0) / 45
            match rot: 
                case 0:
                    self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], 0 * 0.25, 1 * 0.25, width=0.03, length_includes_head=True)
                case 1:
                    self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], 1 * 0.17, 1 * 0.17, width=0.03, length_includes_head=True)
                case 2:
                    self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], 1 * 0.25, 0 * 0.25, width=0.03, length_includes_head=True)
                case 3:
                    self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], 1 * 0.17, -1 * 0.17, width=0.03, length_includes_head=True)
                case 4:
                    self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], 0 * 0.25, -1 * 0.25, width=0.03, length_includes_head=True)
                case 5:
                    self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], -1 * 0.17, -1 * 0.17, width=0.03, length_includes_head=True)
                case 6:
                    self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], -1 * 0.25, 0 * 0.25, width=0.03, length_includes_head=True)
                case 7:
                    self.ax0.arrow(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], -1 * 0.17, 1 * 0.17, width=0.03, length_includes_head=True)
                case _:
                    print('Error: rot bug!')    # For debug

            #self.fig_grid_diplay.savefig(os.path.dirname(__file__) +"/Testing/test_map/test_map{}.png".format(i))
            #plt.ioff()
            #plt.pause(0.00001)

        if save:
            Image.fromarray(event.frame).save(os.path.dirname(__file__)+'/dataset/test_frame{}.jpeg'.format(i), 'jpeg')
            i += 1

        #self.event = self.bc.last_event              # current state information (after action)
        
        if abs(event.metadata['agent']['position']['x'] - self.goal['position']['x']) <= 0.5 and \
           abs(event.metadata['agent']['position']['z'] - self.goal['position']['z']) <= 0.5 and \
           abs(event.metadata['agent']['rotation']['y'] - self.goal['rotation']['y']) == 0 :
            self.reward = 10 
            self.succeed = True
            self.done = True
            if self.test_disp_treja == True:
                plt.ion()
                self.ax0.plot(self.trej_x, self.trej_y, 'bo-')
                plt.ioff()
                plt.pause(0.001)

        if not self.succeed:
            #self.dist = ((event.metadata['agent']['position']['x'] - self.goal['position']['x'])**2 + (event.metadata['agent']['position']['z'] - self.goal['position']['z'])**2)
            self.dist = vector_distance(event.metadata['agent']['position'], self.goal['position'])

            if not event.metadata['lastActionSuccess'] and self.ma == True:  
                self.collided = True
                #self.reward -= 0.2
            else:
                self.collided = False

            # if self.action == 'MoveAhead' and self.collided==False:
            #     self.reward += 0.1*self.ma_rec
            #     if self.last_distance-self.dist > 0:
            #         self.reward += 0.1
            # else:
            # if self.rl_rec>2:
            #     self.reward -= 0.1*self.rl_rec

            # if self.last_distance-self.dist > 0:
            #     self.reward -= 0.01*(self.big_distance-self.dist)
            # if self.last_distance-self.dist < 0:
            #     self.reward += 0.01*(self.big_distance-self.dist)

            self.last_distance = self.dist
        
        if self.step_count >= self.max_step:
            self.done = True
            self.reward = -10
            if self.test_disp_treja == True:
                plt.ion()
                self.ax0.plot(self.trej_x, self.trej_y, 'ro-')
                plt.ioff()
                plt.pause(0.001)

        return torch.Tensor(event.frame.copy()).view(1,3,300,300).cuda().float(), self.collided, self.reward, self.done, self.succeed

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

    def reset_for_testing(self, goal_info, max_step=None, Self_Adjust_task_step_N=None, length_limit=None):
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

        self.length_limit = length_limit
        print('--- Reset Task for testing ---   L>=5 limit: ', self.length_limit)
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
            self.automax_step = Self_Adjust_task_step_N
            self.max_step = max_step
            if self.automax_step is not None:
                self.max_step = self.len_path * self.automax_step  # 依照倍率自動設定每一task MAX step 否則 依固定的task MAX step

            # L>=5 True
            if self.length_limit:
                if (self.len_path >= 5) & (self.len_path < self.max_step):  # 設置正確 結束target設定迴圈
                    print('---- L>=5 length limit ----  L=%s' % self.len_path, '  MAX Step: %s' % self.max_step)
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
                          '  MAX Step: %s' % self.max_step)  # 設置正確 結束target設定迴圈
                    break
        self.cur_img = torch.Tensor(event.frame.copy(), dtype=torch.float32)
        # t = [0., 0., 0., 0., 0., 0.]
        t = [0. for i in range(self.action_dim)]
        # t = [0., 0., 0., 0., 0.]    # N_A 3->5
        self.pre_action = torch.from_numpy(np.array(t, dtype=np.float32))
        return self.cur_img.view(1,3,300,300), self.target_img.view(1,3,300,300), self.len_path, self.pre_action, self.bc.grid_points, self.goalname


if __name__ == '__main__':
    env = ActivateEnv()
    k = 410
    _cur_world = 'FloorPlan' + str(k)
    print(_cur_world)
    env.target_setup_tool(_cur_world)


