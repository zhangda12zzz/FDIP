import numpy as np
import torch

# train = np.load('GGIP/data_all/SingleOne-IMU/Smpl_singleone_imus_final.npy', allow_pickle=True)   # 201541
# test = np.load('GGIP/data_all/SingleOne-IMU/Smpl_singleone_imus_test.npy', allow_pickle=True)   # 40960
# # print(test)


# length_train = []
# length_train_all = 0
# train_final = []

# length_test = []
# length_test_all = 0

# delete = False

# for i in train:
#     # if len(i) > 5500 and not delete:
#     #     afterD = i[480:]
#     #     i = afterD
#     #     delete = True
#     length_train.append(len(i))
#     length_train_all += len(i)
#     train_final.append(i)
        
    
# for i in test:
#     length_test.append(len(i))
#     length_test_all += len(i)
    

# # np.save('GGIP/data_all/SingleOne-IMU/Smpl_singleone_imus_final.npy', train_final)
    
# print(length_train)


test = np.load('GGIP/data_all/Mixamo/mixamoSMPLPos_imu_test.npy', allow_pickle=True)
length_test = []
length_test_all = 0
max = 0
min = 10000

for i in test:
    length_test.append(len(i))
    length_test_all += len(i)
    if len(i) > max:
        max = len(i)
    if len(i) < min:
        min = len(i)
        
print(max)