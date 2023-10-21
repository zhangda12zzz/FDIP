r"""
    Config for paths, joint set, and normalizing scales.
"""
import torch


# datasets (directory names) in AMASS
# e.g., for ACCAD, the path should be `paths.raw_amass_dir/ACCAD/ACCAD/s001/*.npz`
# amass_data = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
#               'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
#               'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust67']
amass_data = ['ACCAD', 'BMLhandball', 'BMLmovi', 'BMLrub', 'CMU', 'DanceDB', 'DFaust', 'EKUT', 'EyesJapanDataset', 
              'HDM05', 'HUMAN4D', 'HumanEva', 'KIT', 'Mosh', 'SFU', 'TotalCapture', 'Transitions', 'PosePrior', 'SSM', 'TCDHands']
amass_data_test_tmp = ['ACCAD','BMLhandball', 'BMLmovi']


class paths:
    raw_amass_dir = 'data/raw/AMASS'      # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    amass_dir = 'data/work/AMASS'         # output path for the synthetic AMASS dataset
    raw_dipimu_dir = 'data/raw/DIP_IMU'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = 'data/work/DIP_IMU'      # output path for the preprocessed DIP-IMU dataset

    # DIP recalculates the SMPL poses for TotalCapture dataset. You should acquire the pose data from the DIP authors.
    raw_totalcapture_dip_dir = 'data/dataset_raw/TotalCapture/DIP_recalculate'  # contain ground-truth SMPL pose (*.pkl)
    raw_totalcapture_official_dir = 'data/dataset_raw/TotalCapture/official'    # contain official gt (S1/acting1/gt_skel_gbl_pos.txt)
    totalcapture_dir = 'data/dataset_work/TotalCapture'          # output path for the preprocessed TotalCapture dataset

    result_dir = 'data/result'                                 # output directory for the evaluation results

    smpl_file = 'data/SMPLmodel/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'              # official SMPL model path
    physics_model_file = 'models/urdfmodels/physics.urdf'      # physics body model path
    plane_file = 'models/urdfmodels/plane.urdf'                # (for debug) path to plane.urdf    Please put plane.obj next to it.
    
    weights_file = 'model/weights/weights.pt'                # network weight file
    weights_file_trial = 'data/weights/trial1210_refine45_1011.pt'
    
    # weights_file_tp = 'data/weights/weights_tp.pt'                # network weight file for transpose
    weights_file_tp = 'data/weights/trial1203_tp_256.pt'
    
    physics_parameter_file = 'physics_parameters.json'   # physics hyperparameters


class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    reduced_pos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]
    # TODO：这里的节点顺序要和graph的对照上

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)
    
    # transpose专用
    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]
    
    graphB = [0,4,5,15,18,19]                           # 6 joints
    # graphP = [0,1,2,3,9,12,13,16,14,17]                 # 10 joints
    graphP = [0,4,5,6,9,15,16,18,17,19]                     # 相邻关节合并、合并到远离根节点的顶点中
    graphJ = [0,1,4,2,5,3,6,9,12,15,13,16,18,14,17,19]  # 16 joints
    graphA = [0,1,4,7,10,2,5,8,11,3,6,9,12,15,13,16,18,20,22,14,17,19,21,23]    # 24 joints
    
    graphA_22ver = [0,1,4,7,10,2,5,8,11,3,6,9,12,15,13,16,18,20,14,17,19,21]
    
    graphJ_noRoot = [1,4,2,5,3,6,9,12,15,13,16,18,14,17,19]
    
    graphJ2Reduce = [0,2,4,1,3,5,6,7,9,12,8,10,13,11,14]# graphJ_noRoot转化为reduce关节顺序的list


vel_scale = 3
# transpose专用
acc_scale = 30

# 应该是一个16*3的list
# 静态关节位移，x左为正，y上为正，z前为正（左上前系）=> x右为正【DIP数据集的x和bvh数据的x正方向是相反的】
smpl_static_offset = torch.div(torch.tensor([
    [0.0, 0.0, 0.0],    # 0-root
    [6.031000, -9.051300, -1.354300],      # 1 R-hips
    [3.168952, -38.481613, -0.484300],     # 2 R-knee
    [-5.858100, -8.227997, -1.766400],       # 3 L-hips
    [-2.981390, -38.775951, 0.803700],       # 4 L-knee
    [-0.443900, 12.440399, -3.838500],       # 5 spine1
    [-0.448800, 13.795601, 2.682000],        # 6 spine2
    [0.226500, 5.603203, 0.285500],        # 7 spine3
    [1.339000, 21.163605, -3.346800],      # 8 neck
    [-1.011300, 8.893707, 5.041000],         # 9 head
    [8.295400, 11.247192, -2.370700],      # 10 R-collar
    [11.315668, 4.702499, -0.847200],      # 11 R-shoulder
    [26.239435, 0.000000, 0.000000],       # 12 R-elbow
    [-7.170200, 11.399994, -1.889800],       # 13 L-collar
    [-12.281406, 4.549484, -1.904600],       # 14 L-shoulder
    [-25.683819, 0.000000, -0.000000]        # 15 L-elbow
]).float(), 100.0)

smpl_bvh_offset = [
    [0.0, 0.0, 0.0],
    [5.858100, -8.227997, -1.766400], #l-hip
    [2.981390, -38.775951, 0.803700],
    [-2.981390, -42.608841, -3.742800],
    [4.105402, -6.028602, 12.204200],
    [-6.031000, -9.051300, -1.354300], #r-hip
    [-3.168952, -38.481613, -0.484300],
    [3.168952, -41.928219, -3.456200],
    [-3.484000, -6.210600, 13.032299],
    [0.443900, 12.440399, -3.838500], # spine1
    [0.448800, 13.795601, 2.682000],
    [-0.226500, 5.603203, 0.285500],
    [-1.339000, 21.163605, -3.346800],
    [1.011300, 8.893707, 5.041000], #head
    [7.170200, 11.399994, -1.889800], #l-collar
    [12.281406, 4.549484, -1.904600],
    [25.683819, 0.000000, -0.000000],
    [26.611454, 0.000000, -0.000000],
    [8.648361, -1.481339, -1.314458],
    [-8.295400, 11.247192, -2.370700], #r-collar
    [-11.315668, 4.702499, -0.847200],
    [-26.239435, 0.000000, 0.000000],
    [-26.926125, 0.000000, 0.000000],
    [-8.871140, -1.091568, -0.808778]


]
