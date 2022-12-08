# 0, 8, 16, 180, 328
# total 336

import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import os

# save total traj
path = "/Testing"
list1 = [0, 599]
vid_id = 0
j = 0
# for i in range(1, 6):
#     tar = cv2.imread(path+"test_target/test_target"+str(list1[i-1])+".png")
#     cur = cv2.imread(path+"test_frame/test_frame"+str(list1[i]-1)+".png")
#     maze = cv2.imread(path + "test_traj/traj_" + str(i) + ".png")
#     # BGR2RGB
#     tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
#     cur = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB)
#     maze = cv2.cvtColor(maze, cv2.COLOR_BGR2RGB)
#
#     plt.figure(figsize=(25, 7))
#     plt.suptitle("Navigation", fontsize=25)
#     plt.subplot(1, 3, 1), plt.title("Target image", fontsize=15)
#     plt.imshow(tar), plt.axis('off')
#
#     plt.subplot(1, 3, 2), plt.title("Current image", fontsize=15)
#     plt.imshow(cur), plt.axis('off')
#
#     plt.subplot(1, 3, 3), plt.title("Trajectory", fontsize=15)
#     plt.imshow(maze), plt.axis('off')
#
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig("/home/kert/NavProject/Testing/concat_img/vid" + str(i-1) + "_" + str(list1[i]-list1[i-1]) + ".png")
#     plt.cla()
# # save total traj


# # save concat img
# for i in range(601):
#     print("current i = " + str(i))
#     if i < list1[vid_id + 1]:
#         tar = cv2.imread("/home/kert/NavProject/Testing/test_target/test_target"+str(list1[vid_id])+".png")
#         cur = cv2.imread("/home/kert/NavProject/Testing/test_frame/test_frame"+str(i)+".png")
#         maze = cv2.imread("/home/kert/NavProject/Testing/test_map/test_map"+str(i)+".png")
#         # BGR2RGB
#         tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
#         cur = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB)
#         maze = cv2.cvtColor(maze, cv2.COLOR_BGR2RGB)
#
#
#         plt.figure(figsize=(25,7))
#         plt.suptitle("Navigation", fontsize=25)
#         plt.subplot(1,3,1), plt.title("Target image", fontsize=15)
#         plt.imshow(tar), plt.axis('off')
#
#         plt.subplot(1,3,2), plt.title("Current image", fontsize=15)
#         plt.imshow(cur), plt.axis('off')
#
#         plt.subplot(1,3,3), plt.title("Trajectory", fontsize=15)
#         plt.imshow(maze), plt.axis('off')
#
#         plt.tight_layout()
#         # plt.show()
#         plt.savefig("/home/kert/NavProject/Testing/concat_img/vid" + str(vid_id) + "_" + str(j) + ".png")
#         plt.cla()
#         j+=1
#     else :
#         vid_id += 1
#         j = 0
#         tar = cv2.imread("/home/kert/NavProject/Testing/test_target/test_target"+str(list1[vid_id])+".png")
#         cur = cv2.imread("/home/kert/NavProject/Testing/test_frame/test_frame"+str(i)+".png")
#         maze = cv2.imread("/home/kert/NavProject/Testing/test_map/test_map"+str(i)+".png")
#         # BGR2RGB
#         tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
#         cur = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB)
#         maze = cv2.cvtColor(maze, cv2.COLOR_BGR2RGB)
#
#         plt.figure(figsize=(25,7))
#         plt.suptitle("Navigation", fontsize=25)
#         plt.subplot(1,3,1), plt.title("Target image", fontsize=15)
#         plt.imshow(tar), plt.axis('off')
#
#         plt.subplot(1,3,2), plt.title("Current image", fontsize=15)
#         plt.imshow(cur), plt.axis('off')
#
#         plt.subplot(1,3,3), plt.title("Trajectory", fontsize=15)
#         plt.imshow(maze), plt.axis('off')
#
#         plt.tight_layout()
#         # plt.show()
#         plt.savefig("/home/kert/NavProject/Testing/concat_img/vid" + str(vid_id) + "_" + str(j) + ".png")
#         plt.cla()
#         j+=1
# # save concat img

# videolize
# file will apear in Testing (same floder as python file)
path = os.getcwd()+"/Testing/test_frame/"
# 0~8, 0~8, 0~164, 0~148, 0~9
result_name = "Navigation_303_REAL_5act_new_reward_Rotate_Reward更正_MA改001(10fps).mp4"
frame_list = []
for i in range(0,28322):
    frame_list.append(path + "test_frame" + str(i) + ".png")

fps = 10  # 2 or 10
shape = cv2.imread(frame_list[0]).shape # delete dimension 3
size = (shape[1], shape[0])
print("frame size: ",size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(result_name, fourcc, fps, size)

for idx, path in enumerate(frame_list):
    frame = cv2.imread(path)
    # print("\rMaking videos: {}/{}".format(idx+1, len(frame_list)), end = "")
    current_frame = idx+1
    total_frame_count = len(frame_list)
    percentage = int(current_frame*30 / (total_frame_count+1))
    print("\rProcess: [{}{}] {:06d} / {:06d}".format("#"*percentage, "."*(30-1-percentage), current_frame, total_frame_count), end ='')
    out.write(frame)

out.release()
print("Finish making video !!!")
# videolize




# print("frame count: ",len(frame_list))
# # 0, 8, 16, 180, 328
# fps = 30
# shape = cv2.imread(frame_list[0]).shape # delete dimension 3
# size = (shape[1], shape[0])
# print("frame size: ",size)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(result_name, fourcc, fps, size)
#
# for idx, path in enumerate(frame_list):
#     frame = cv2.imread(path)
#     # print("\rMaking videos: {}/{}".format(idx+1, len(frame_list)), end = "")
#     current_frame = idx+1
#     total_frame_count = len(frame_list)
#     percentage = int(current_frame*30 / (total_frame_count+1))
#     print("\rProcess: [{}{}] {:06d} / {:06d}".format("#"*percentage, "."*(30-1-percentage), current_frame, total_frame_count), end ='')
#     out.write(frame)
#
# out.release()
# print("Finish making video !!!")
