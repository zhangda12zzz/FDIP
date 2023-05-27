import os
import numpy as np
import torch
from torch import optim

from model.utils import GAN_loss, ImagePool, get_ee, Criterion_EE, Eval_Criterion, Criterion_EE_2
from model.base_model import BaseModel
from model.integrated_CIP import IntegratedModelCIP, testCIP
from model.integrated import IntegratedModel
from option_parser import try_mkdir
from utils.Quaternions import Quaternions


class GAN_model_CIP(BaseModel):
    def __init__(self, args, dataset, std_paths=None, log_path=None):
        super(GAN_model_CIP, self).__init__(args, log_path=log_path)
        self.character_names = ['Smpl']
        self.dataset = dataset
        # self.models = []
        # self.D_para = []    # 判别器参数
        # self.G_para = []    # 生成器参数
        self.args = args
        self.std_path = std_paths
        self.epochCount = 0

        # 构建一个融合模型（生成器+判别器）！！这是重点
        # modelTest = testCIP(args)
        
        # TODO:用于人体姿态估计的版本
        # model_CIP = IntegratedModelCIP(args, dataset.joint_topology, None, self.device, self.character_names)
        # TODO：用于人体姿态优化的版本
        model_CIP = IntegratedModel(args, dataset.joint_topology, None, self.device, self.character_names)
        self.models = model_CIP
        self.D_para = model_CIP.D_parameters()
        self.G_para = model_CIP.G_parameters()

        self.criterion_rec = torch.nn.MSELoss()
        if self.is_train:   # 训练阶段，为D和G构建优化器、损失函数
            # self.fake_pools = []
            self.optimizerD = optim.Adam(self.D_para, args.learning_rate / 10.0, betas=(0.9, 0.999))
            self.optimizerG = optim.Adam(self.G_para, args.learning_rate, betas=(0.9, 0.999))
            self.optimizers = [self.optimizerD, self.optimizerG]
            self.criterion_gan = GAN_loss(args.gan_mode).to(self.device)
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_ee = Criterion_EE(args, torch.nn.MSELoss())
            self.fake_pools = ImagePool(args.pool_size)
        else:       # 如果是测试阶段，则需要为产生的结果bvh文件设置存储路径balabala
            import option_parser
            self.err_crit = []
            self.err_crit = Eval_Criterion(dataset.joint_topology)
            self.id_test = 0
            self.bvh_path = os.path.join(args.save_dir, 'results/bvh')
            option_parser.try_mkdir(self.bvh_path)

        from dataset.bvh_writer import BVH_writer
        from dataset.bvh_parser import BVH_file
        import option_parser
        if self.std_path:
            file = BVH_file(self.std_path)
        else:
            file = BVH_file(option_parser.get_std_bvh(dataset='Smpl'))
        self.bvhWriter = BVH_writer(file.edges, file.names)

    def set_input(self, motions):
        # motions_input 的尺寸：[n,(4+3)*5,t_w]
        self.motions_input = motions    # 包括了[2, motion, character], 其中、motion:tensor[1,C,t_w], character:一个int

        # if not self.is_train:   # ？？？
        #     self.motion_backup = []
        #     # for i in range(self.n_topology): 
        #     self.motion_backup = motions.clone()
        #     self.motions_input[0][1:] = self.motions_input[0][0]        # 全部变成静态动作
        #     self.motions_input[1] = [0] * len(self.motions_input[1])    # 全部默认0号模型

    def discriminator_requires_grad_(self, requires_grad):
        # for model in self.models:
        model = self.models
        for para in model.discriminator.parameters():
            para.requires_grad = requires_grad

    def forward(self):
        self.epochCount += 1
        self.offset_repr = self.models.static_encoder(self.dataset.offsets)  # [n,v(22),3] + [n,6v'(72)] + [n,12v'(144)]

        # reconstruct ======:
        offsets = [self.offset_repr[p] for p in range(self.args.num_layers + 1)] # 提取该拓扑的offset
        
        # TODO: 姿态估计用，imu作为输入
        # imu, motion = self.motions_input
        # motion = motion.to(self.device)
        # imu = imu.to(self.device)
        # self.motions = motion
        # self.imudetects = imu
        # motion_denorm = self.dataset.denorm_motion(motion)
        # self.motion_denorm = motion_denorm
        # imu_denorm = self.dataset.denorm_imuData(imu)
        # self.imu_denorm = imu_denorm
        # TODO: 姿态优化用，motion作为输入
        imu_denorm, motion = self.motions_input
        imu = self.dataset.norm_motion(imu_denorm)
        
        motion_denorm = self.dataset.denorm_motion(motion)
        self.motions = motion
        self.motion_denorm = motion_denorm

        latent, res = self.models.auto_encoder(imu, offsets)  # 编码后的动作[n, 4v''*4, t_w/4] + 再解码后的动作[n,4v-4+3, t_w]
        
        # TODO: 姿态优化用，要求完全对照
        # res[:,0:4] = motion[:,0:4]
        res_denorm = self.dataset.denorm_motion(res)                                            # 重建姿势：逆标准化
        # res_denorm[:,-3:] = torch.zeros(res_denorm.shape[0], 3, res_denorm.shape[2])
        
        res_pos = self.models.fk.forward_from_raw(res_denorm, self.dataset.offsets)   # 重建姿势：求根节点坐标
        self.res_pos = res_pos
        self.latents = latent
        self.res = res
        self.res_denorm = res_denorm
        
        height = self.models.height
        height = height.reshape((height.shape[0], 1, height.shape[1], 1))
        
        res_ee = get_ee(res_pos, self.dataset.joint_topology, self.dataset.ee_ids,
                    velo=self.args.ee_velo, from_root=self.args.ee_from_root)
        res_ee /= height
        self.res_ee = res_ee
        
        pos = self.models.fk.forward_from_raw(motion_denorm, self.dataset.offsets).detach()   # 输入姿势：根节点坐标
        ee = get_ee(pos, self.dataset.joint_topology, self.dataset.ee_ids,
                    velo=self.args.ee_velo, from_root=self.args.ee_from_root)
        ee /= height
        self.pos_ref = pos    # target根节点位置（intra gt）
        self.ee_ref = ee      # target末端执行器
        gt_latent = self.models.auto_encoder.enc(motion, offsets)
        self.latents_ref = gt_latent


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        GAN网络中判别器D的反向传播！
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())     # 通过detach断开了反向传播链，所以前面生成器的参数不会更新！
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_Ds = []
        self.loss_D = 0
        """
        A->A
        """
        # for i in range(self.n_topology):
        fake = self.fake_pools.query(self.res_pos)
        self.loss_Ds.append(self.backward_D_basic(self.models.discriminator, self.pos_ref.detach(), fake))
        self.loss_D += self.loss_Ds[-1]
        self.loss_recoder.add_scalar('D_loss', self.loss_Ds[-1])

    def backward_G(self):
        r'''生成器计算损失 & 反向传播'''
        #rec_loss and gan loss
        self.rec_losses = []
        self.rec_loss = 0
        self.cycle_loss = 0
        self.loss_G = 0
        self.ee_loss = 0
        self.loss_G_total = 0
        
        # 重建损失 L_rec
        # for i in range(self.n_topology):
        rec_loss1 = self.criterion_rec(self.motions[:,4:-3], self.res[:,4:-3])
        self.loss_recoder.add_scalar('rec_loss_quater', rec_loss1)

        height = self.models.real_height
        height = height.reshape(height.shape + (1, 1,))
        input_pos = self.motion_denorm[:, -3:, :] / height
        rec_pos = self.res_denorm[:, -3:, :] / height
        rec_loss2 = self.criterion_rec(input_pos, rec_pos)
        self.loss_recoder.add_scalar('rec_loss_global', rec_loss2)

        pos_ref_global = self.models.fk.from_local_to_world(self.pos_ref) / height.reshape(height.shape + (1, ))
        res_pos_global = self.models.fk.from_local_to_world(self.res_pos) / height.reshape(height.shape + (1, ))
        rec_loss3 = self.criterion_rec(pos_ref_global, res_pos_global)
        self.loss_recoder.add_scalar('rec_loss_position', rec_loss3)

        rec_loss = rec_loss1 + (rec_loss2 * self.args.lambda_global_pose +
                                rec_loss3 * self.args.lambda_position) * 100

        self.rec_losses = rec_loss
        self.rec_loss += rec_loss

        # for src in range(self.n_topology):
        #     for dst in range(self.n_topology):
        # 潜在一致性损失L_ltc
        cycle_loss = self.criterion_cycle(self.latents_ref, self.latents)
        self.loss_recoder.add_scalar('cycle_loss', cycle_loss)
        self.cycle_loss += cycle_loss
        # 末端执行器损失L_ee
        ee_loss = self.criterion_ee(self.ee_ref, self.res_ee)
        self.loss_recoder.add_scalar('ee_loss', ee_loss)
        self.ee_loss += ee_loss
        
        # GAN损失，应该指的是输出的判别结果与标注（只训练生成器时标注为True）之间的损失差 -> 就是L_adv
        if self.args.gan_mode != 'none':
            loss_G = self.criterion_gan(self.models.discriminator(self.res_pos), True)
        else:
            loss_G = torch.tensor(0)
        self.loss_recoder.add_scalar('G_loss', loss_G)
        self.loss_G += loss_G

        self.loss_G_total = self.rec_loss * self.args.lambda_rec + \
                            self.ee_loss * self.args.lambda_ee / 2 +\
                            self.loss_G * 1 +\
                            self.cycle_loss * self.args.lambda_cycle / 2
        self.loss_G_total.backward()        # 反向传播

    def optimize_parameters(self):
        r'''正向传播+反向传播的过程'''
        self.forward()

        # update Gs
        # 先更新生成器的参数
        self.discriminator_requires_grad_(False)    # 停止判别器的参数更新
        self.optimizerG.zero_grad()
        self.backward_G()                           # 计算论文中提到的4项损失，进行反向推导
        self.optimizerG.step()

        # update Ds
        # 再更新判别器的参数
        if self.args.gan_mode != 'none':# and self.epochCount % 2 == 0:
            self.discriminator_requires_grad_(True) # 开始判别器参数更新（并没有停止生成器参数的更新？）
            self.optimizerD.zero_grad()
            self.backward_D()
            self.optimizerD.step()
        else:
            self.loss_D = torch.tensor(0)   # 不使用GAN，则不训练、也不更新判别器的参数

    def verbose(self):
        res = {'rec_loss': self.rec_losses.item(),
               'cycle_loss': self.cycle_loss.item(),
               'ee_loss': self.ee_loss.item(),
               'D_loss_gan': self.loss_D.item(),
               'G_loss_gan': self.loss_G.item(),
               }
        return sorted(res.items(), key=lambda x: x[0])

    def save(self, suffix=None):
        if suffix:
            self.model_save_dir = str(suffix)
        # for i, model in enumerate(self.models):
        self.models.save(os.path.join(self.model_save_dir, 'topology'), self.epoch_cnt)

        for i, optimizer in enumerate(self.optimizers):
            file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(self.epoch_cnt, i))
            try_mkdir(os.path.split(file_name)[0])
            torch.save(optimizer.state_dict(), file_name)

    def load(self, epoch=None, suffix=None):
        # for i, model in enumerate(self.models):
        if suffix:
            self.model_save_dir = str(suffix)
            
        self.models.load(os.path.join(self.model_save_dir, 'topology'), epoch)

        if self.is_train:
            for i, optimizer in enumerate(self.optimizers):
                file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(epoch, i))
                optimizer.load_state_dict(torch.load(file_name))
        self.epoch_cnt = epoch

    def compute_test_result(self):
        gt_poses = []
        gt_denorm = []
        for src in range(self.n_topology):
            gt = self.motion_backup[src]
            idx = list(range(gt.shape[0]))
            gt = self.dataset.denorm(src, idx, gt)
            gt_denorm.append(gt)
            gt_pose = self.models[src].fk.forward_from_raw(gt, self.dataset.offsets[src][idx])
            gt_poses.append(gt_pose)
            for i in idx:
                new_path = os.path.join(self.bvh_path, self.character_names[src][i])
                from option_parser import try_mkdir
                try_mkdir(new_path)
                self.bvhWriter[src][i].write_raw(gt[i, ...], 'quaternion', os.path.join(new_path, '{}_gt.bvh'.format(self.id_test)))

        p = 0
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                for i in range(len(self.character_names[dst])):
                    dst_path = os.path.join(self.bvh_path, self.character_names[dst][i])
                    self.bvhWriter[dst][i].write_raw(self.fake_res_denorm[p][i, ...], 'quaternion',
                                                  os.path.join(dst_path, '{}_{}.bvh'.format(self.id_test, src)))
                p += 1

        self.id_test += 1



    def SMPLtest(self, motions_input, storeBvh=False):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            imu_denorm, motion = motions_input  # [n,C(4v-4+3),t_w(64)],  一个int对应character的序号
            motion = motion.to(self.device)
            motion_denorm = self.dataset.denorm_motion(motion)
            imu_denorm = imu_denorm.to(self.device)
            
            # TODO: 姿态优化用，此时输入的imu和motion就分别是 前一个motion和后一个motion，二者的尺寸相同
            imu = self.dataset.norm_motion(imu_denorm)
            
            res = self.testForward(imu)
            # res[:,0:4] = motion[:,0:4]
            res_denorm = self.dataset.denorm_motion(res) 
            # res_denorm[:,-3:] = torch.zeros(res_denorm.shape[0], 3, res_denorm.shape[2])
            
            imu_pos = self.models.fk.forward_from_raw(imu_denorm, self.dataset.offsets)
            res_pos = self.models.fk.forward_from_raw(res_denorm, self.dataset.offsets)
            motion_pos = self.models.fk.forward_from_raw(motion_denorm, self.dataset.offsets)
            loss = self.testLoss(motion, res, motion_pos, res_pos) # quat + global
            loss2 = self.testLoss(motion, imu, motion_pos, imu_pos)
            
            res_fullRot = self.comput_fullRotate(res_denorm)
            gt_fullRot = self.comput_fullRotate(motion_denorm)
            if storeBvh:
                self.compute_SMPLtestResult(motion_denorm, res_denorm)
            rec_loss = self.criterion_rec(res_fullRot, gt_fullRot)
            if self.args.is_train:
                self.loss_recoder.add_scalar('rec_loss_POSAll_val', rec_loss)
            return res_fullRot, gt_fullRot, loss, loss2
        
    def testForward(self, input):
        self.offset_repr = self.models.static_encoder(self.dataset.offsets)
        offsets = [self.offset_repr[p] for p in range(self.args.num_layers + 1)] # 提取该拓扑的offset
        
        imu_denorm = input  # 采用标准化的数据输入
        # imu_denorm = self.dataset.denorm_imuData(input) #[1,42,t] # 采用去标准化的原本数据输入网络，CIP，CIPe2e都是这样
        latent, res = self.models.auto_encoder(imu_denorm, offsets)  # 编码后的动作[n, 4v''*4, t_w/4] + 再解码后的动作[n,4v-4+3, t_w]
        return res
    
    def testLoss(self, motion_gt, motion_res, pos_gt, pos_res):
        # 重建损失 L_rec
        # for i in range(self.n_topology):
        rec_loss1 = self.criterion_rec(motion_gt, motion_res)
        if self.args.is_train:
            self.loss_recoder.add_scalar('rec_loss_quater_val', rec_loss1)

        height = self.models.real_height
        height = height.reshape(height.shape + (1, 1,))

        pos_ref_global = self.models.fk.from_local_to_world(pos_gt) / height.reshape(height.shape + (1, ))
        res_pos_global = self.models.fk.from_local_to_world(pos_res) / height.reshape(height.shape + (1, ))
        rec_loss3 = self.criterion_rec(pos_ref_global, res_pos_global)
        if self.args.is_train:
            self.loss_recoder.add_scalar('rec_loss_position_val', rec_loss3 * 100)

        return rec_loss1, (rec_loss3 * self.args.lambda_position) * 100

        # 末端执行器损失L_ee
        # ee_loss = self.criterion_ee(self.ee_ref, self.res_ee)
        # # self.loss_recoder.add_scalar('ee_loss', ee_loss)
        # self.ee_loss += ee_loss
        
    def comput_fullRotate(self, motion_denorm): # [1,87,t]
        # res_pose = self.models.fk.forward_from_raw(motion_denorm, self.dataset.offsets).squeeze(0)
        res_motions = motion_denorm.permute(0,2,1).detach().cpu().numpy()
        rotations_full_all = []
        for res_motion in res_motions:
            # positions = motion[:, -3:]
            rotations = res_motion[:, :-3].reshape((res_motion.shape[0], -1, 4))
            norm = rotations[:, :, 0] ** 2 + rotations[:, :, 1] ** 2 + rotations[:, :, 2] ** 2 + rotations[:, :, 3] ** 2
            norm = np.repeat(norm[:, :, np.newaxis], 4, axis=2)
            rotations /= norm
            rotations_quat = Quaternions(rotations)
            rotations = rotations_quat.transforms()
            rotations_full = np.zeros((rotations.shape[0], self.bvhWriter.joint_num, 3, 3))
            # 看一下self.bvhWriter.names定一下顺序
            for idx, edge in enumerate(self.bvhWriter.edge2joint):
                if edge != -1:
                    rotations_full[:, idx] = rotations[:, edge]
            rotations_full_all.append(rotations_full)
        return torch.Tensor(rotations_full)
 
    def compute_SMPLtestResult(self, motion_denorm, res_denorm):
        r'''
            返回计算出的所有retarting结果
            在1类拓扑（只有0/1两类）有多个模型的情况下，返回[n,C(),t]
            一般来说只有一个模型，返回的就是[1,C(87),t]
        '''
        gt = motion_denorm
        gt_pose = self.models.fk.forward_from_raw(gt, self.dataset.offsets)
        res = res_denorm
        res_pose = self.models.fk.forward_from_raw(res, self.dataset.offsets)
        idx = list(range(gt.shape[0]))
        for i in idx:
            new_path = os.path.join(self.bvh_path, self.character_names[0])
            self.bvhWriter.write_raw(gt[i, ...], 'quaternion', os.path.join(new_path, '{}_CIP_gt.bvh'.format(self.id_test)))
            self.bvhWriter.write_raw(res[i, ...], 'quaternion', os.path.join(new_path, '{}_CIP_pre.bvh'.format(self.id_test)))
        self.id_test += 1
        