from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from CSA_VAE import VAE
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
bc=Controller(scene='FloorPlan303',
             #platform=CloudRendering,
             snapToGrid=True,
             width=256,
             height=256,
             fieldOfView=90)
pos = bc.step(dict(action='GetReachablePositions'))
#print(pos.metadata["actionReturn"])
#print(pos.metadata["actionReturn"])

# count = 0
# for i in pos.metadata["actionReturn"]:
#         x = i['x']
#         y = i['y']
#         z = i['z']
#         for r in [45, 90, 135, 180, 225, 270, 315, 360]:
#             ev = bc.step(
#                 action="Teleport",
#                 position=dict(x=x, y=y, z=z),
#                 rotation=dict(x=0, y=r, z=0)
#             )
#             img = ev.frame
#             print(np.size(ev.frame))
#             #print(ev.metadata["agent"])
#             img = Image.fromarray(img, 'RGB')
#             img.save('dataset2/{}.jpeg'.format(count))
#             print('dataset2/{}.jpeg'.format(count))
#             count += 1

event = bc.step(action='RotateRight', degrees=270)

frame = event.frame



encoder_net = VAE(64)
encoder_net.to('cuda')                
state_dict = torch.load('VAE_64z_batch128.pt', map_location=lambda storage, loc: storage.cuda())

encoder_net.load_state_dict(state_dict)
encoder_net.eval()

_, _, _, predicted = encoder_net.forward(torch.from_numpy(event.frame.copy()/255).float().view(1,3,256,256).cuda())

print(predicted)

plt.imshow(np.clip(predicted.squeeze().permute(1, 2, 0).cpu().detach().numpy(), 0, 1))
plt.show()

input("Press Enter to continue...")
