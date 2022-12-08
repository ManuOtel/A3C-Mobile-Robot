from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
bc=Controller(scene='FloorPlan303',
             platform=CloudRendering,
             snapToGrid=True,
             width=300,
             height=300,
             fieldOfView=90)
pos = bc.step(dict(action='GetReachablePositions'))
print(pos.metadata["actionReturn"])