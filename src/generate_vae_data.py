"""A3C Visual Navigation Mobile Robot project.

This is a script used to generate data for the VAE.

@Author: Emanuel-Ionut Otel
@University: National Chung Cheng University
@Created: 2022-06-13
@Links: www.ccu.edu.tw
@Contact: manuotel@gmail.com
"""

#### ---- IMPORTS AREA ---- ####
from ai2thor.controller import Controller
import matplotlib.pyplot as plt
from PIL import Image
import keyboard
#### ---- IMPORTS AREA ---- ####

#### ----GLOBAL INIT AREA ---- ####
bc=Controller(scene='FloorPlan303',
              gridSize=0.25,
              snapToGrid=True,
              width=256,
              height=256,
              fieldOfView=90,
              renderObjectImage=False)
#### ----GLOBAL INIT AREA ---- ####


def save_data(pos, count:int=0):
    """This function is used to creat and save the data, for VAE training.

    :param pos: The reachable positions.
    :param count: The number of images saved.

    :return: None
    """
    for i in pos.metadata["actionReturn"]:
            x = i['x']
            y = i['y']
            z = i['z']
            for r in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
                ev = bc.step(
                    action="Teleport",
                    position=dict(x=x, y=y, z=z),
                    rotation=dict(x=0, y=r, z=0)
                )
                img = ev.frame
                img = Image.fromarray(img, 'RGB')
                img.save('dataset5/{}.jpeg'.format(count))
                print('dataset5/{}.jpeg'.format(count))
                count += 1


def goal_frame_test() -> None:
    """This function is used to test the goal frame.

    :return: None
    """
    event = bc.step(action='RotateRight', degrees=270)
    reach_pos = bc.step(dict(action='GetReachablePositions'))
    ev = None
    for obj in event.metadata["objects"]:
        print(obj['objectType'])
        if obj["objectType"] == 'Desk':
            tx = obj['position']['x']
            ty = event.metadata['agent']['position']['y']
            tz = obj['position']['z']
            ry = round(ty)
            minn = float('inf')
            maxx = float('-inf')
            for i in reach_pos.metadata["actionReturn"]:
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
    ev = bc.step(
        action="Teleport",
        position=dict(x=aux, y=ty, z=auz),
        rotation=dict(x=0, y=ry, z=0),
        )
    # bc.step(action='RotateLeft', degrees=180)
    # bc.step(action='MoveAhead')
    event = bc.step(action='RotateLeft', degrees=90)

    plt.imshow(event.frame)
    plt.show()


def keyboard_control(step_pass:bool=True) -> None:
    """This function is used to control the robot with the keyboard.
    
    :param step_pass: If True, the enviroment will execute a dummy step after each action.
    
    :return: None"""
    while True:  
        event = None
        if keyboard.is_pressed('w'):  # if key 'q' is pressed 
            event = bc.step(action='MoveAhead')
            print(event)
            if step_pass:
                bc.step('Pass')
        if keyboard.is_pressed('a'):  # if key 'q' is pressed 
            event = bc.step(action='RotateLeft', degrees=90)
            print(event)
            if step_pass:
                bc.step('Pass')
        if keyboard.is_pressed('d'):  # if key 'q' is pressed 
            event = bc.step(action='RotateRight', degrees=90)
            print(event)
            if step_pass:
                bc.step('Pass')
        if keyboard.is_pressed('space'):  # if key 'q' is pressed 
            event = bc.step(action='RotateRight', degrees=90)
            print(event)
            if step_pass:
                bc.step('Pass')
            plt.imshow(event.frame)
            plt.show()
            

if __name__=="__main__":
    print('Testing')
