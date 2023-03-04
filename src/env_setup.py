"""A3C Visual Navigation Mobile Robot project.

This is the module used to setup the AI2-THOR environment.

@Author: Emanuel-Ionut Otel
@University: National Chung Cheng University
@Created: 2022-06-13
@Links: www.ccu.edu.tw
@Contact: manuotel@gmail.com
"""


#### ---- IMPORTS AREA ---- ####
import warnings
warnings.filterwarnings("ignore")
import random, cv2, os, torch, gc
import numpy as np
import matplotlib.pyplot as plt
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ai2thor.util.metrics import path_distance, vector_distance
from PIL import Image
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
#### ---- IMPORTS AREA ---- ####


#### ---- GLOBAL INIT AREA ---- ####
global i
i = 0
#### ---- GLOBAL INIT AREA ---- ####



def set_all_seeds(seed:int) -> None:
    """Sets all seeds for reproducibility.
    
    param: seed: The seed to set.
    
    return: None"""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ActivateEnv():
    """This class is used to setup the AI2-THOR environment.
    
    param: action_space: The action space to use.
    param: length_limit: If True, the robot will not be allowed to spawn close to the target.
    param: max_step: The maximum number of steps the robot can take.
    param: automax_step: If True, the maximum number of steps will be automatically set, based on the startup position.
    param: cloud_rendering: If True, the environment will be rendered in the cloud.
    """
    def __init__(self, action_space, 
                 length_limit:bool=False, 
                 max_step=None, 
                 automax_step=None,
                 cloud_rendering:bool=False):
        self.action_space = action_space
        self.action_dim = len(action_space)
        self.cur_world = 'FloorPlan303'
        self.dim = 256
        if cloud_rendering:
            self.bc = Controller(scene=self.cur_world,
                                 platform=CloudRendering,
                                 snapToGrid=False,
                                 width=self.dim,
                                 height=self.dim,
                                 fieldOfView=90)
        else:
            self.bc = Controller(scene=self.cur_world,
                                 snapToGrid=False,
                                 width=self.dim,
                                 height=self.dim,
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
        self.col_rec = 0
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
        self.goal_list = ['Desk', 'ShelvingUnit', 'Window', 'Mirror']
        self.goal_angels = {'Desk':90, 'ShelvingUnit':180, 'Window':180, 'Mirror':0}
        self.rotations = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        self.set_goal()
        print('Enviroment succesfuly initiated!')


    def CheckAllGoal(self, _cur_world) -> None:
        """This function is used to check if all the goals are in the graph.
        
        param: _cur_world: The current world.
        
        return: None"""
        self._cur_world = _cur_world
        print(self._cur_world)
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


    def target_setup_tool(self, _cur_world) -> None:
        """This function is used to setup the target.

        param: _cur_world: The current world.

        return: None"""
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
        """This function is used to set the target randomly.

        param: goalkeys: The list of the goals.

        return: target: The target goal."""
        if not self.target_res:                 # if self.target_res =[]
            self.target_res = list(goalkeys.items())
            random.shuffle(self.target_res)     # 直接執行 不用變數再存
        target = random.choice(self.target_res)
        self.target_res.remove(target)
        return target


    def set_goal(self) -> None:
        """This function is used to set the goal randomly, automatically.

        param: None

        return: None"""
        self.goal_name = random.choice(self.goal_list)
        self.goal_angle = self.goal_angels[self.goal_name]
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
        #get_shortest_path_to_point
        #self.bc.step(action='RotateLeft', degrees=90)
        #self.bc.step(action="MoveAhead")
        ev = self.bc.step(action='RotateLeft', degrees=self.goal_angle)
        self.goal['frame']=ev.frame
        self.goal['position']=ev.metadata['agent']['position']
        self.goal['rotation']=ev.metadata['agent']['rotation']

        self.big_distance = vector_distance(self.goal['position'], dict(x=maux, y=y, z=mauz))

        gc.collect()


    def reset(self):
        """This function is used to reset the environment.

        param: None

        return: A tuple with the current image, target image, optimal path, and goal name."""
        self.dist = 0
        self.last_distance = 0
        self.step_count = 0
        self.event = self.bc.reset()    
        self.set_goal()
        self.reset_num += 1
        self.frame = 0
        self.reward = 0

        #self.bc.start_search(scene=self._cur_world)

        while True:
            position = random.choice(self.reach_pos.metadata['actionReturn'])
            #print(position)
            rotation = random.choice(self.rotations)
            self.bc.step(action="Teleport",
                          position=position,
                          rotation=dict(x=0, y=rotation, z=0),
                          horizon=0,
                          standing=True)
            #self.bc.step('Pass')

            # event = self.bc.step(action="GetShortestPathToPoint",
            #                           position=position,
            #                           x=self.goal['position']["x"],
            #                           y=self.goal['position']["y"],
            #                           z=self.goal['position']["z"])
            event = self.bc.step(action="GetShortestPathToPoint",
                                position=position,
                                target=self.goal['position'])

            self.reward = 0
            try:
                self.optimal_path = event.metadata["actionReturn"]['corners']
                self.optimal_path_length = path_distance(self.optimal_path)
            except:
                self.reward = -1

            # Self Adjust MAX Step
            if self.automax_step is not None:
                self.max_step = int(self.optimal_path_length * self.automax_step)

            # L>=5 True
            if self.length_limit and self.reward==0:
                if (self.optimal_path_length > 1):
                    break
            else:
                if abs(event.metadata['agent']['rotation']['y'] - self.goal['rotation']['y']) > 45 and (self.optimal_path_length > 0.5) and self.reward==0:
                    break

        self.cur_img=torch.from_numpy(event.frame.copy()).float()
        self.target_img=torch.from_numpy(self.goal['frame'].copy()).float()
        #print(self.cur_img)
        gc.collect()
        return self.cur_img.view(1,3,self.dim,self.dim).cuda(), self.target_img.view(1,3,self.dim,self.dim).cuda(), self.optimal_path_length, self.goal_name


    def step(self, idx, save=False):
        """This function is used to step the environment.

        param: idx: The index of the action.
        param: save: Whether to save the image of the current view. (usefull for creating a dataset)
        
        return: A tuple with the current image, collision check, success check, and reward."""
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
                #self.bc.step('Pass')
                self.ma = True
                self.ma_rec += 1 
                self.rl_rec = 0
            case 'RotateRight':
                event = self.bc.step(action="RotateRight", degrees=45)
                #event = self.bc.step('Pass')
                self.ma_rec = 0
                self.rl_rec += 1
                #self.reward -= 0.1
            case 'RotateRight90':
                event = self.bc.step(action='RotateRight', degrees=90)
                #self.bc.step('Pass')
                self.ma_rec = 0
                self.rl_rec += 1
            case 'RotateLeft':
                event = self.bc.step(action="RotateLeft", degrees=45)
                #self.bc.step('Pass')
                self.ma_rec = 0
                self.rl_rec += 1
                #self.reward -= 0.1
            case 'RotateLeft90':
                event = self.bc.step(action="RotateLeft", degrees=90)
                #self.bc.step('Pass')
                self.ma_rec = 0
                self.rl_rec += 1
            case 'RotateBack':
                event = self.bc.step(action="RotateLeft", degrees=180)
                #self.bc.step('Pass')
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
           abs(event.metadata['agent']['rotation']['y'] - self.goal['rotation']['y']) < 45 :
            self.reward = 1000
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
                self.col_rec += 1
                self.reward -= 0.03*self.col_rec
            elif self.ma == True:
                self.col_rec = 0
                self.collided = False
                self.reward += 0.01

            if self.rl_rec>1:
                self.reward -= 0.03*self.rl_rec

            self.reward -= 0.01

            if self.last_distance-self.dist > 0:
                self.reward += 0.01
            if self.last_distance-self.dist < 0:
                self.reward -= 0.01#*(self.big_distance-self.dist)

            # self.reward += (self.last_distance - self.dist)*0.1

            self.last_distance = self.dist
        
        if self.step_count >= self.max_step:
            self.done = True
            self.reward = -90
            if self.test_disp_treja == True:
                plt.ion()
                self.ax0.plot(self.trej_x, self.trej_y, 'ro-')
                plt.ioff()
                plt.pause(0.001)

        return torch.Tensor(event.frame.copy()).view(1,3,self.dim,self.dim).cuda().float(), self.collided, self.reward, self.done, self.succeed

if __name__ == '__main__':
    env = ActivateEnv()
    k = 410
    _cur_world = 'FloorPlan' + str(k)
    print(_cur_world)
    env.target_setup_tool(_cur_world)


