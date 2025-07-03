
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # debug专用
import sys
import time

import torch
from torch.utils.data.dataloader import DataLoader

from model.architecture import GAN_model_GIP
from data.dataset_poseReg import ImuMotionData
import option_parser
from option_parser import try_mkdir
import articulate as art


class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator('data/SMPLmodel/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', joint_mask=torch.tensor([1, 2, 16, 17]))

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
    device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
    args.dataset = 'Smpl'
    args.device = device
    
    evaluator = PoseEvaluator()

    # log_path = os.path.join(args.save_dir, 'logs_CIP/') # './pretrained/logs/'
    try_mkdir(args.save_dir)
    # try_mkdir(log_path)

    with open(os.path.join(args.save_dir, 'para.txt'), 'w') as para_file:
        para_file.write(' '.join(sys.argv))     # 存储相关参数

    dataset = ImuMotionData(args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # model = GAN_model_GIP(args, dataset, log_path='train/logs/gaip_uniformM')
    model = GAN_model_GIP(args, dataset, epoch_k=6, log_path='train/logs/expe_pretrainCompare/GAIP/doublefinetune')
   # if args.epoch_begin:
    #     model.load(epoch=args.epoch_begin, download=False)
    # model.load(epoch=500, suffix='GGIP/checkpoints/trial9_gan_all123_ganpose', loadOpt=False)
    
    
    # model.load(epoch=200, suffix='GGIP/checkpoints/gaip_uniformM')
    # model.load(epoch=50, suffix='GGIP/checkpoints/gaip_v1')
    # args.epoch_begin = 101
    # model.load(epoch=100, suffix='train/checkpoints/expe_pretrainCompare/GAIP/posLoss')
    args.epoch_begin = 301
    model.load(epoch=300, suffix='train/checkpoints/expe_pretrainCompare/GAIP/k6')
    
    # 只加载判别器
    # model.models.discriminator.load_state_dict(torch.load(os.path.join('GGIP/checkpoints/trial9_gan_all123_ganpose/topology/500', 'discriminator.pt'),
    #                                                  map_location=device))

    model.setup()

    start_time = time.time()

    to15Joints = [1,2,7,12,3, 8,13,15,16,19, 24,20,25,21,26]    # 按照smpl原本的标准关节顺序定义的15个躯干节点
    reduced = [0,1,2,3,4, 5,6,9,12,13, 14,16,17,18,19]          # 没有头，但是包含了根节点
    sip_err = 0
    ang_err = 0
    jerk_err = 0

    for epoch in range(args.epoch_begin, args.epoch_num):
        model.models.pose_encoder.train() # GAIP
        # model.models.auto_encoder.train() # transpose & PIP
        for step, data in enumerate(data_loader):    # motion: [n,42,64]+[n,87,64]        
            model.set_input(data)
            model.optimize_parameters()
            
            if args.verbose:
                res = model.verbose()
                print('[{}/{}]\t[{}/{}]\t'.format(epoch, args.epoch_num, step, len(data_loader)), res)

        # if epoch > 25 and (epoch % 25 == 0 or epoch == args.epoch_num - 1):
        model.save('train/checkpoints/expe_pretrainCompare/GAIP/doublefinetune')
            
        if epoch % 50 == 0:
            imu, joint, pose, root, gt24 = dataset.getValData()
            # imu, joint, pose, root, gt24 = dataset.getValData()
            offline_errs = []
            test_loss = []
            model.models.pose_encoder.eval()   
            for i in range(imu.shape[0]):
                test_data = [imu[i:i+1], joint[i:i+1], pose[i:i+1], root[i:i+1]]
                # test_data = [imu[i:i+1], joint[i:i+1], pose[i:i+1], root[i:i+1]]
                # gt = gt24[i:i+1].to(device)
                loss, gt_pose, pre_pose = model.SMPLtest(test_data)
                test_loss.append(loss)
                
                if epoch % 100 == 0:
                    offline_errs.append(evaluator.eval(pre_pose, gt_pose))   # 比较经过四元数变换的数据是否和原本一样，结论：一样
            
            if epoch % 100 == 0:
                offline_err = torch.stack(offline_errs).mean(dim=0)
                evaluator.print(offline_err)
                sip_err = offline_err[0,0]
                ang_err = offline_err[1,0]
                jerk_err = offline_err[4,0]
            print('test_loss:', torch.stack(test_loss).mean())
        else:
            model.loss_recoder.add_scalar('rec_loss_r6d/val', model.rec_loss_val_store)
        
                
        model.loss_recoder.add_scalar('SIP_err', sip_err)
        model.loss_recoder.add_scalar('Ang_err', ang_err)
        model.loss_recoder.add_scalar('Jerk_err', jerk_err)
        # else:
        #     model.loss_recoder.add_scalar('SIP_err', 0)
        #     model.loss_recoder.add_scalar('Ang_err', 0)
            
        model.epoch()

    end_tiem = time.time()
    print('training time', end_tiem - start_time)


if __name__ == '__main__':
    main()
    # 查看tensornoard数据：tensorboard --logdir=./path/to/the/folder --port 8123
