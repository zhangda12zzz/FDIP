import sys
import numpy as np
import pickle
import torch
import h5py

def display_pkl_data(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
            print(f"Data in {file_path}:")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"Key: {key}, Shape: {value.shape}")
                    print(f"  First element: {value[0]}")
                elif isinstance(value, list):
                    print(f"Key: {key}, Type: list, Length: {len(value)}")
                    if len(value) > 0 and isinstance(value[0], np.ndarray):
                        print(f"  First element shape: {value[0].shape}")
                        print(f"  First element: {value[0]}")
                    if len(value) > 0 and isinstance(value[0], (np.ndarray, list, int, float, str)):
                        print(f" Len:{len(value[1])}     First （3）few elements: {value[:3]}")
                else:
                    print(f"Key: {key}, Type: {type(value)}")
                    print(f"  Value: {value}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except pickle.UnpicklingError:
        print(f"Error: The file {file_path} is not a valid pickle file.")
    except Exception as e:
        print(f"An error occurred: {e}")


import numpy as np

def display_np_data(file_path):
    try:
        data = np.load(file_path,allow_pickle=True)
        print(f"Data in {file_path}:")

        # 处理 .npz 文件（多个数组）
        if isinstance(data, np.lib.npyio.NpzFile):
            for key in data.keys():
                value = data[key]
                print(f"Key: {key}, Shape: {value.shape}")

                # 添加维度检查
                if value.ndim == 0:  # 处理标量
                    print(f"  Value: {value.item()}")
                elif value.size > 0:  # 处理可索引数据
                    print(f"  First element: {value[0]}")
                else:
                    print("  Empty array")

                print(f"  First element: {value[0]}")

        # 处理 .npy 文件（单个数组）
        elif isinstance(data, np.ndarray):
            print(f"Shape: {data.shape}，Type: {data.dtype}")
            print(f"  First element: {data[0]}")
            print(f"  First element  shape: {data[0].shape}，Type: {data[0].dtype}")

        else:
            print(f"Unsupported data type: {type(data)}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def display_pt_data(file_path):
    try:
        data = torch.load(file_path)
        print(f"Data in {file_path}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"Key: {key}")
                if isinstance(value, torch.Tensor):
                    print(f"  Type: torch.Tensor, Shape: {value.shape}")
                    print(f"  First element: {value[0]}")
                elif isinstance(value, list):
                    print(f"  Type: list")
                    if len(value) > 0 and isinstance(value[0], torch.Tensor):
                        print(f" len(value) :{len(value)} First element shape: {value[0].shape}")
                    else:
                        print(f"  Value: {value}")
                else:
                    print(f"  Type: {type(value)}")
                    print(f"  Value: {value}")
        else:
            print(f"Type: {type(data)}")
            if isinstance(data, torch.Tensor):
                print(f"  Shape: {data.shape}")
            else:
                print(f"  Ka_Gaip:   len:{len(data)}   value[0]: {data[0].size()},first element: {data[0]}")
                #print(f"  Value: {data}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def display_pth_data(file_path):
    try:
        data = torch.load(file_path,)
        print(f"Data in {file_path}:")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"Key: {key}, Shape: {value.shape}")
                    print(f"  First element: {value[0]}")
                else:
                    print(f"Key: {key}, Type: {type(value)}")
                    print(f"  Value: {value}")
        else:
            print(f"Type: {type(data)}")
            print(f"  Value: {data}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def display_hdf5_data(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            print(f"Data in {file_path}:")
            def print_attrs(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {obj.shape}")
                    print(f"  First element: {obj[0]}")
                elif isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
            file.visititems(print_attrs)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    """  pkl
    """
    #pkl_file_path = r"F:\CodeForPaper\Dataset\TotalCapture_Real_60FPS\s2_acting1.pkl"    # (3386, 6, 3)   gt: (3385, 72)  (3386, 6, 3, 3)
    #pkl_file_path = r"F:\CodeForPaper\Dataset\RELI11D_Dataset\train\RELI_train_p1.pkl"     #图像中的2D坐标
    #pkl_file_path = r"F:\CodeForPaper\Dataset\DIPIMUandOthers\DIP_IMU\DIP_IMU\s_01\01.pkl"   #gt、sip、sop:(13778, 72)   acc：(13778, 17, 3)  IMu：(13778, 17, 12)  ori：  (13778, 17, 3, 3)
    #pkl_file_path = (r"F:\CodeForPaper\Dataset\HEva\S1_Box_1.pkl")   #acc:(1370, 6, 3) pose:(1370,135)     ori:(1370,6,3,3)
    #pkl_file_path = r"F:\CodeForPaper\dip18-master\data_synthesis\Jog_1.pkl"   #原始合成数据之前

    #pkl_file_path = r"F:\CodeForPaper\dip18-master\data_synthesis\Jog_1_synthesis.pkl"   #合成之后的数据集
    #beta:10   gender：female    poses：（1365,72）   ori：(1365,6,3,3)   acc:(1365,6,3)


    #display_pkl_data(pkl_file_path)

    """  npz、npy
    """

    #npz_file_path = r"F:\CodeForPaper\Dataset\SingleOne\processed\Smpl_singleone_motion_SMPL24_test.npy"   # (25,866,24,3,3)
    #npz_file_path = r"F:\CodeForPaper\Dataset\SingleOne\processed\Smpl_singleone_imus_test.npy"    # (25,866,72)    加速度18+54旋转矩阵

    #npz_file_path = r"F:\CodeForPaper\Dataset\mixamo\mixamoSMPLPos_imu_test.npy"   # imu参数  18+54  (949,136,72)
    #npz_file_path = r"F:\CodeForPaper\Dataset\mixamo\mixamoSMPLPos_motion_SMPL24_test.npy"  #  旋转矩阵真值 (949，136,24,3,3)

    #npz_file_path = r"F:\CodeForPaper\Dataset\DIPIMUandOthers\DIP_IMU_nn\imu_own_training.npz"   # (949,136,72)
    # npz_file_path = r"F:\CodeForPaper\Dataset\mixamo\mixamoSMPLPos_motion_SMPL24_test.npy"  # (949，136,24,3,3)

    #display_np_data(npz_file_path)

    """  pt
    """
    #pt_file_path = r"F:\CodeForPaper\Dataset\DIPIMUandOthers\DIP_6\Detail\vacc.pt"
    # pt_file_path = r"F:\CodeForPaper\Dataset\SingleOne\Pt\tran.pt" #vrot：旋转矩阵-测量姿态   pose：真值
    #Type: <class 'list'>
    #Ka_Gaip:   len:60   value[0]: torch.Size([13766, 24, 3, 3])

    pt_file_path = r"F:\CodeForPaper\Dataset\AMASS\HumanEva\pt\joint.pt"
    # Type: <class 'list'>
    #   Ka_Gaip:   len:28   value[0]: torch.Size([1372, 6, 3])

    # pt_file_path = r"F:\CodeForPaper\Dataset\DIPIMUandOthers\DIP_6\test.tpt"     #acc:(60,13766,6,3)    ori:(60,13766,6,3,3)    poses:(60,13766,72)  tran:(60,13766,3)

    #pt_file_path = r"F:\CodeForPaper\Dataset\TotalCapture_Real_60FPS\KaPt\test.pt"
    # Key: acc
    #   Type: list
    #  len(value) :45 First element shape: torch.Size([4113, 6, 3])

    #pt_file_path = r"F:\CodeForPaper\Dataset\TotalCapture_Real_60FPS\KaPt\split_actions\pose.pt"



    display_pt_data(pt_file_path)
    """
    每个序列长度不一致
    """




    """  pth、hdf5
    """
    # #
    # pth_file_path = r"path_to_your_pth_file.pth"
    # display_pth_data(pth_file_path)
    #
    # hdf5_file_path = r"path_to_your_hdf5_file.hdf5"
    # display_hdf5_data(hdf5_file_path)
