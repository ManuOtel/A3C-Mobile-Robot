"""A3C Visual Navigation Mobile Robot project.

This script is used to create a a functional top-view of the agent inside AI2Thor enviroment.

@Author: Emanuel-Ionut Otel
@University: National Chung Cheng University
@Created: 2022-06-13
@Links: www.ccu.edu.tw
@Contact: manuotel@gmail.com
"""


#### ---- IMPORTS AREA ---- ####
import matplotlib, keyboard, copy, math
import numpy as np
import matplotlib.pyplot as plt
from ai2thor.controller import Controller
from PIL import Image, ImageDraw
#### ---- IMPORTS AREA ---- ####


#### ---- GLOBAL INIT AREA ---- ####
matplotlib.use("TkAgg")
#### ---- GLOBAL INIT AREA ---- ####


class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )


def position_to_tuple(position):
    """Converts a position dictionary to a tuple of (x, y, z).
    
    :param position: A position dictionary.
    
    :return: A tuple of (x, y, z)."""
    return (position["x"], position["y"], position["z"])


def get_agent_map_data(c: Controller) -> dict:
    """Returns the agent map data.

    :param c: The controller.

    :return: The agent map data."""
    c.step({"action": "ToggleMapView"})
    cam_position = c.last_event.metadata["cameraPosition"]
    cam_orth_size = c.last_event.metadata["cameraOrthSize"]
    pos_translator = ThorPositionTo2DFrameTranslator(c.last_event.frame.shape, position_to_tuple(cam_position), cam_orth_size)
    to_return = {
        "frame": c.last_event.frame,
        "cam_position": cam_position,
        "cam_orth_size": cam_orth_size,
        "pos_translator": pos_translator,
    }
    c.step({"action": "ToggleMapView"})
    return to_return


def add_agent_view_triangle(position, rotation, frame, pos_translator, scale=1.0, opacity=0.7) -> np.ndarray:
    """Adds a triangle with the agent view to the map.

    :param position: The agent position.
    :param rotation: The agent rotation.
    :param frame: The frame.
    :param pos_translator: The position translator.
    :param scale: The scale.
    :param opacity: The opacity.

    :return: The new frame.
    """
    p0 = np.array((position[0], position[2]))
    p1 = copy.copy(p0)
    p2 = copy.copy(p0)

    theta = -2 * math.pi * (rotation / 360.0)
    rotation_mat = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    offset1 = scale * np.array([-1, 1]) * math.sqrt(2) / 2
    offset2 = scale * np.array([1, 1]) * math.sqrt(2) / 2

    p1 += np.matmul(rotation_mat, offset1)
    p2 += np.matmul(rotation_mat, offset2)

    img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

    opacity = int(round(255 * opacity))  # Define transparency for the triangle.
    points = [tuple(reversed(pos_translator(p))) for p in [p0, p1, p2]]
    draw = ImageDraw.Draw(img2)
    draw.polygon(points, fill=(255, 255, 255, opacity))

    img = Image.alpha_composite(img1, img2)
    return np.array(img.convert("RGB"))


if __name__ == "__main__":
    
    c = Controller(scene='FloorPlan1_physics',
                   gridSize=0.25,
                   snapToGrid=False,
                   width=256,
                   height=256,
                   fieldOfView=90,
                   renderObjectImage=True)
    
    c.step(dict(action='Initialize', gridSize=0.25))
    c.reset("FloorPlan1_physics")

    # pos = bc.step(dict(action='GetReachablePositions'))
    # pos.metadata["actionReturn"]:
    
    t = get_agent_map_data(c)
    new_frame = add_agent_view_triangle(
        position_to_tuple(c.last_event.metadata["agent"]["position"]),
        c.last_event.metadata["agent"]["rotation"]["y"],
        t["frame"],
        t["pos_translator"],
    )

    while True:  # making a loop
        if keyboard.is_pressed('w'):  # if key 'w' is pressed 
            event = c.step(action='MoveAhead')
            print(event)
            c.step('Pass')
            t = get_agent_map_data(c)
            new_frame = add_agent_view_triangle(
                position_to_tuple(c.last_event.metadata["agent"]["position"]),
                c.last_event.metadata["agent"]["rotation"]["y"],
                t["frame"],
                t["pos_translator"],
            )
        if keyboard.is_pressed('a'):  # if key 'a' is pressed 
            event = c.step(action='RotateLeft', degrees=45)
            print(event)
            c.step('Pass')
            t = get_agent_map_data(c)
            new_frame = add_agent_view_triangle(
                position_to_tuple(c.last_event.metadata["agent"]["position"]),
                c.last_event.metadata["agent"]["rotation"]["y"],
                t["frame"],
                t["pos_translator"],
            )
        if keyboard.is_pressed('d'):  # if key 'd' is pressed 
            event = c.step(action='RotateRight', degrees=45)
            print(event)
            c.step('Pass')
            t = get_agent_map_data(c)
            new_frame = add_agent_view_triangle(
                position_to_tuple(c.last_event.metadata["agent"]["position"]),
                c.last_event.metadata["agent"]["rotation"]["y"],
                t["frame"],
                t["pos_translator"],
            )
        