import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用
import sys
import time
import torch
from torch.utils.data.dataloader import DataLoader

from model import create_model, create_CIPmodel
from dataset import create_dataset_CIP, get_character_names
import option_parser
from option_parser import try_mkdir
from model.tp_rnn import TransPoseNet
from config import paths
import articulate as art

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator('articulate/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        r'''
        返回【蒙面关节全局角度误差】【关节全局角度误差】【关节位置错误】【顶点位置错误】*100 【预测运动抖动】/100
        '''
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


def main():
    args = option_parser.get_args()
    args.dataset = 'Smpl'
    device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
    # characters = get_character_names(args)  # 两组模型，对应两种拓扑 [['Smpl'], ['Aj']]
    evaluator = PoseEvaluator()

    log_path = os.path.join(args.save_dir, 'logs_CIP/') # './pretrained/logs/'
    try_mkdir(args.save_dir)
    try_mkdir(log_path)

    with open(os.path.join(args.save_dir, 'para.txt'), 'w') as para_file:
        para_file.write(' '.join(sys.argv))     # 存储相关参数

    dataset = create_dataset_CIP(args, std_paths='dataset/CIP/work/CIP_std_22.bvh')
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = create_CIPmodel(args, dataset, log_path=log_path, std_paths='dataset/CIP/work/CIP_std_22.bvh')
    if args.epoch_begin:
        model.load(epoch=args.epoch_begin, download=False)
    # model.load(epoch=1400, suffix='pretrained/models_CIP')

    model.setup()

    start_time = time.time()

    net = TransPoseNet(num_past_frame=20, num_future_frame=5).to(device)
    net.load_state_dict(torch.load(paths.weights_file))
    net.eval()
    
    to15Joints = [1,2,7,12,3, 8,13,15,16,19, 24,20,25,21,26]    # 按照smpl原本的标准关节顺序定义的15个躯干节点
    reduced = [0,1,2,3,4, 5,6,9,12,13, 14,16,17,18,19]          # 没有头，但是包含了根节点

    for epoch in range(args.epoch_begin, args.epoch_num):
        for step, motions in enumerate(data_loader):    # motion: [n,42,64]+[n,87,64]
            imu = motions[0]
            imu = imu.permute(0,2,1).to(device)
            _, motion_preprocess = net.calculatePose(imu)
            motion_preprocess = torch.Tensor(motion_preprocess).permute(0,2,1).to(device)
            
            motion = motions[1].to(device)
            # motion_denorm = model.dataset.denorm_motion(motion)
            input = [motion_preprocess, motion]
            
            model.set_input(input)
            model.optimize_parameters()

            if args.verbose:
                res = model.verbose()
                print('[{}/{}]\t[{}/{}]\t'.format(epoch, args.epoch_num, step, len(data_loader)), res)

        if epoch % 20 == 0 or epoch == args.epoch_num - 1:
            # model.save('pretrained/models_CIP')
            model.save('pretrained/models_CIP_refine')
            
        test_imu, test_motion = dataset.getValData()
        test_imu = test_imu.permute(0,2,1).to(device)
        _, test_input_motion = net.calculatePose(test_imu)
        test_input_motion = torch.Tensor(test_input_motion).permute(0,2,1).to(device)
        test_data = [test_input_motion,test_motion]
        res_fullrot, gt_fullrot, loss, loss2 = model.SMPLtest(test_data)
        print('val_loss: {}'.format(loss))

        model.epoch()

    end_tiem = time.time()
    print('training time', end_tiem - start_time)


if __name__ == '__main__':
    main()
