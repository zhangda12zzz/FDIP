import os
import torch
from torch import optim

from model.utils import GAN_loss, ImagePool, get_ee, Criterion_EE, Eval_Criterion, Criterion_EE_2
from model.base_model import BaseModel
from option_parser import try_mkdir
from model.integrated import IntegratedModel


class GAN_model(BaseModel):
    def __init__(self, args, character_names, dataset, std_paths=None):
        super(GAN_model, self).__init__(args)
        self.character_names = character_names  # 2组
        self.dataset = dataset  # 2组
        self.n_topology = len(character_names)
        self.models = []
        self.D_para = []    # 判别器参数
        self.G_para = []    # 生成器参数
        self.args = args
        self.std_path = std_paths

        for i in range(self.n_topology):    # 对于每组模型（2种拓扑）
            # 构建一个融合模型（生成器+判别器）！！这是重点
            if std_paths:
                model = IntegratedModel(args, dataset.joint_topologies[i], None, self.device, character_names[i], self.std_path[i])
            else:
                model = IntegratedModel(args, dataset.joint_topologies[i], None, self.device, character_names[i])
            self.models.append(model)
            self.D_para += model.D_parameters()
            self.G_para += model.G_parameters()

        if self.is_train:   # 训练阶段，为D和G构建优化器、损失函数
            self.fake_pools = []
            self.optimizerD = optim.Adam(self.D_para, args.learning_rate, betas=(0.9, 0.999))
            self.optimizerG = optim.Adam(self.G_para, args.learning_rate, betas=(0.9, 0.999))
            self.optimizers = [self.optimizerD, self.optimizerG]
            self.criterion_rec = torch.nn.MSELoss()
            self.criterion_gan = GAN_loss(args.gan_mode).to(self.device)
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_ee = Criterion_EE(args, torch.nn.MSELoss())
            for i in range(self.n_topology):
                self.fake_pools.append(ImagePool(args.pool_size))
        else:       # 如果是测试阶段，则需要为产生的结果bvh文件设置存储路径balabala
            import option_parser
            self.err_crit = []
            for i in range(self.n_topology):
                self.err_crit.append(Eval_Criterion(dataset.joint_topologies[i]))
            self.id_test = 0
            self.bvh_path = os.path.join(args.save_dir, 'results/bvh')
            option_parser.try_mkdir(self.bvh_path)

            self.writer = []
            for i in range(self.n_topology):
                writer_group = []
                for j, char in enumerate(character_names[i]):
                    from dataset.bvh_writer import BVH_writer
                    from dataset.bvh_parser import BVH_file
                    import option_parser
                    if self.std_path:
                        file = BVH_file(self.std_path[i][j])
                    else:
                        file = BVH_file(option_parser.get_std_bvh(dataset=char))
                    # file = BVH_file(option_parser.get_std_bvh(dataset=char))
                    writer_group.append(BVH_writer(file.edges, file.names))
                self.writer.append(writer_group)

    def set_input(self, motions):
        self.motions_input = motions    # 包括了[2, motion, character], 其中、motion:tensor[1,C,t_w], character:一个int

        if not self.is_train:
            self.motion_backup = []
            for i in range(self.n_topology):
                self.motion_backup.append(motions[i][0].clone())
                self.motions_input[i][0][1:] = self.motions_input[i][0][0]
                self.motions_input[i][1] = [0] * len(self.motions_input[i][1])

    def discriminator_requires_grad_(self, requires_grad):
        for model in self.models:
            for para in model.discriminator.parameters():
                para.requires_grad = requires_grad

    def forward(self):
        self.latents = []
        self.offset_repr = []
        self.pos_ref = []
        self.ee_ref = []
        self.res = []
        self.res_denorm = []
        self.res_pos = []
        self.fake_res = []
        self.fake_res_denorm = []
        self.fake_pos = []
        self.fake_ee = []
        self.fake_latent = []
        self.motions = []
        self.motion_denorm = []
        self.rnd_idx = []

        for i in range(self.n_topology):    # self.dataset.offsets[i]): [n,v,3]
            self.offset_repr.append(self.models[i].static_encoder(self.dataset.offsets[i]))  # [n,v,3] + [n,6v'] + [n,12v']

        # reconstruct
        for i in range(self.n_topology):
            motion, offset_idx = self.motions_input[i]  # [n,C(4v-4+3),t_w(64)],  一个int对应character的序号
            motion = motion.to(self.device)
            self.motions.append(motion)                     # self.motions: 包含2个[n,C(4v-4+3),t_w(64)]的list 【<<1个】

            motion_denorm = self.dataset.denorm(i, offset_idx, motion)  # 对应character的序号对motion去标准化
            self.motion_denorm.append(motion_denorm)        # self.motion_denorm: 包含2个[n,C(4v-4+3),t_w(64)]的list 【<<2个】
            offsets = [self.offset_repr[i][p][offset_idx] for p in range(self.args.num_layers + 1)] # 提取该拓扑的offset
            latent, res = self.models[i].auto_encoder(motion, offsets)  # 编码后的动作[n, 4v''*4, t_w/4] + 再解码后的动作[n,4v-4+3, t_w]
            res_denorm = self.dataset.denorm(i, offset_idx, res)                                            # 重建姿势：逆标准化
            res_pos = self.models[i].fk.forward_from_raw(res_denorm, self.dataset.offsets[i][offset_idx])   # 重建姿势：求根节点坐标
            self.res_pos.append(res_pos)
            self.latents.append(latent)
            self.res.append(res)
            self.res_denorm.append(res_denorm)

            pos = self.models[i].fk.forward_from_raw(motion_denorm, self.dataset.offsets[i][offset_idx]).detach()   # 输入姿势：根节点坐标
            ee = get_ee(pos, self.dataset.joint_topologies[i], self.dataset.ee_ids[i],
                        velo=self.args.ee_velo, from_root=self.args.ee_from_root)
            height = self.models[i].height[offset_idx]
            height = height.reshape((height.shape[0], 1, height.shape[1], 1))
            ee /= height
            self.pos_ref.append(pos)    # target根节点位置（intra gt）
            self.ee_ref.append(ee)      # target末端执行器

        # retargeting
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                if self.is_train:
                    rnd_idx = torch.randint(len(self.character_names[dst]), (self.latents[src].shape[0],))
                else:
                    rnd_idx = list(range(self.latents[0].shape[0]))
                self.rnd_idx.append(rnd_idx)
                dst_offsets_repr = [self.offset_repr[dst][p][rnd_idx] for p in range(self.args.num_layers + 1)]
                # 将编码后的动作self.latents，通过另一套骨骼offset解码，之后和重建流程类似，构建新模型新动作的pos、ee、height等数据
                fake_res = self.models[dst].auto_encoder.dec(self.latents[src], dst_offsets_repr)
                fake_latent = self.models[dst].auto_encoder.enc(fake_res, dst_offsets_repr)

                fake_res_denorm = self.dataset.denorm(dst, rnd_idx, fake_res)
                fake_pos = self.models[dst].fk.forward_from_raw(fake_res_denorm, self.dataset.offsets[dst][rnd_idx])
                fake_ee = get_ee(fake_pos, self.dataset.joint_topologies[dst], self.dataset.ee_ids[dst],
                                 velo=self.args.ee_velo, from_root=self.args.ee_from_root)
                height = self.models[dst].height[rnd_idx]
                height = height.reshape((height.shape[0], 1, height.shape[1], 1))
                fake_ee = fake_ee / height

                self.fake_latent.append(fake_latent)
                self.fake_pos.append(fake_pos)
                self.fake_res.append(fake_res)
                self.fake_ee.append(fake_ee)
                self.fake_res_denorm.append(fake_res_denorm)


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
        A->A, A->B, B->A, B->B@[0, 1, 2, 3]
        """
        for i in range(self.n_topology):
            fake = self.fake_pools[i].query(self.fake_pos[2 - i])
            self.loss_Ds.append(self.backward_D_basic(self.models[i].discriminator, self.pos_ref[i].detach(), fake))
            self.loss_D += self.loss_Ds[-1]
            self.loss_recoder.add_scalar('D_loss_{}'.format(i), self.loss_Ds[-1])

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
        for i in range(self.n_topology):
            rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('rec_loss_quater_{}'.format(i), rec_loss1)

            height = self.models[i].real_height[self.motions_input[i][1]]
            height = height.reshape(height.shape + (1, 1,))
            input_pos = self.motion_denorm[i][:, -3:, :] / height
            rec_pos = self.res_denorm[i][:, -3:, :] / height
            rec_loss2 = self.criterion_rec(input_pos, rec_pos)
            self.loss_recoder.add_scalar('rec_loss_global_{}'.format(i), rec_loss2)

            pos_ref_global = self.models[i].fk.from_local_to_world(self.pos_ref[i]) / height.reshape(height.shape + (1, ))
            res_pos_global = self.models[i].fk.from_local_to_world(self.res_pos[i]) / height.reshape(height.shape + (1, ))
            rec_loss3 = self.criterion_rec(pos_ref_global, res_pos_global)
            self.loss_recoder.add_scalar('rec_loss_position_{}'.format(i), rec_loss3)

            rec_loss = rec_loss1 + (rec_loss2 * self.args.lambda_global_pose +
                                    rec_loss3 * self.args.lambda_position) * 100

            self.rec_losses.append(rec_loss)
            self.rec_loss += rec_loss

        p = 0
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                # 潜在一致性损失L_ltc
                cycle_loss = self.criterion_cycle(self.latents[src], self.fake_latent[p])
                self.loss_recoder.add_scalar('cycle_loss_{}_{}'.format(src, dst), cycle_loss)
                self.cycle_loss += cycle_loss
                # 末端执行器损失L_ee
                ee_loss = self.criterion_ee(self.ee_ref[src], self.fake_ee[p])
                self.loss_recoder.add_scalar('ee_loss_{}_{}'.format(src, dst), ee_loss)
                self.ee_loss += ee_loss
                # GAN损失，应该指的是输出的判别结果与标注（只训练生成器时标注为True）之间的损失差 -> 就是L_adv
                if src != dst:
                    if self.args.gan_mode != 'none':
                        loss_G = self.criterion_gan(self.models[dst].discriminator(self.fake_pos[p]), True)
                    else:
                        loss_G = torch.tensor(0)
                    self.loss_recoder.add_scalar('G_loss_{}_{}'.format(src, dst), loss_G)
                    self.loss_G += loss_G

                p += 1

        self.loss_G_total = self.rec_loss * self.args.lambda_rec  + \
                            self.cycle_loss * self.args.lambda_cycle / 2 + \
                            self.ee_loss * self.args.lambda_ee / 2 + \
                            self.loss_G * 1
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
        if self.args.gan_mode != 'none':
            self.discriminator_requires_grad_(True) # 开始判别器参数更新（并没有停止生成器参数的更新？）
            self.optimizerD.zero_grad()
            self.backward_D()
            self.optimizerD.step()
        else:
            self.loss_D = torch.tensor(0)   # 不使用GAN，则不训练、也不更新判别器的参数

    def verbose(self):
        res = {'rec_loss_0': self.rec_losses[0].item(),
               'rec_loss_1': self.rec_losses[1].item(),
               'cycle_loss': self.cycle_loss.item(),
               'ee_loss': self.ee_loss.item(),
               'D_loss_gan': self.loss_D.item(),
               'G_loss_gan': self.loss_G.item()}
        return sorted(res.items(), key=lambda x: x[0])

    def save(self, suffix=None):
        if suffix:
            self.model_save_dir = self.model_save_dir + str(suffix)
        for i, model in enumerate(self.models):
            model.save(os.path.join(self.model_save_dir, 'topology{}'.format(i)), self.epoch_cnt)

        for i, optimizer in enumerate(self.optimizers):
            file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(self.epoch_cnt, i))
            try_mkdir(os.path.split(file_name)[0])
            torch.save(optimizer.state_dict(), file_name)

    def load(self, epoch=None):
        for i, model in enumerate(self.models):
            model.load(os.path.join(self.model_save_dir, 'topology{}'.format(i)), epoch)

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
                self.writer[src][i].write_raw(gt[i, ...], 'quaternion', os.path.join(new_path, '{}_gt.bvh'.format(self.id_test)))

        p = 0
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                for i in range(len(self.character_names[dst])):
                    dst_path = os.path.join(self.bvh_path, self.character_names[dst][i])
                    self.writer[dst][i].write_raw(self.fake_res_denorm[p][i, ...], 'quaternion',
                                                  os.path.join(dst_path, '{}_{}.bvh'.format(self.id_test, src)))
                p += 1

        self.id_test += 1



    def SMPLtest(self, motion_input=None, offset_idx=None):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            # self.retargetForward()
            # self.computeResult()
            return self.retargetForward2(motion_input, offset_idx)
        
    def retargetForward(self):
        # self.latents = []
        self.offset_repr = []
        # self.pos_ref = []
        # self.ee_ref = []
        # self.res = []
        # self.res_denorm = []
        # self.res_pos = []
        # self.fake_res = []
        # self.fake_res_denorm = []
        # self.fake_pos = []
        # self.fake_ee = []
        # self.fake_latent = []
        # self.motions = []
        # self.motion_denorm = []
        # self.rnd_idx = []
        
        for i in range(self.n_topology):    # self.dataset.offsets[i]): [n,v,3]
            self.offset_repr.append(self.models[i].static_encoder(self.dataset.offsets[i]))  # [n,v,3] + [n,6v'] + [n,12v']

        # 重建过程，构造独立于骨骼的动作表示
        motion, offset_idx = self.motions_input[0]  # [n,C(4v-4+3),t_w(64)],  一个int对应character的序号
        motion = motion.to(self.device)
        # self.motions.append(motion)                     # self.motions: 包含2个[n,C(4v-4+3),t_w(64)]的list 【<<1个】

        # motion_denorm = self.dataset.denorm(i, offset_idx, motion)  # 对应character的序号对motion去标准化
        # self.motion_denorm.append(motion_denorm)        # self.motion_denorm: 包含2个[n,C(4v-4+3),t_w(64)]的list 【<<2个】
        offsets = [self.offset_repr[0][p][offset_idx] for p in range(self.args.num_layers + 1)] # 提取该拓扑的offset
        latent, res = self.models[0].auto_encoder(motion, offsets)  # 编码后的动作[n, 4v''*4, t_w/4] + 再解码后的动作[n,4v-4+3, t_w]
        # res_denorm = self.dataset.denorm(i, offset_idx, res)                                            # 重建姿势：逆标准化
        # res_pos = self.models[0].fk.forward_from_raw(res_denorm, self.dataset.offsets[0][offset_idx])   # 重建姿势：求根节点坐标
        # self.res_pos.append(res_pos)
        # self.latents.append(latent)
        # self.res.append(res)
        # self.res_denorm.append(res_denorm)

        src = 0
        dst = 1
        rnd_idx = list(range(latent.shape[0]))
        dst_offsets_repr = [self.offset_repr[dst][p][rnd_idx] for p in range(self.args.num_layers + 1)]
        # 将编码后的动作self.latents，通过另一套骨骼offset解码，之后和重建流程类似，构建新模型新动作的pos、ee、height等数据
        # fake_res = self.models[dst].auto_encoder.dec(self.latents[src], dst_offsets_repr)
        fake_res = self.models[dst].auto_encoder.dec(latent, dst_offsets_repr)
        # fake_latent = self.models[dst].auto_encoder.enc(fake_res, dst_offsets_repr)

        fake_res_denorm = self.dataset.denorm(dst, rnd_idx, fake_res)
        # fake_pos = self.models[dst].fk.forward_from_raw(fake_res_denorm, self.dataset.offsets[dst][rnd_idx])
        # fake_ee = get_ee(fake_pos, self.dataset.joint_topologies[dst], self.dataset.ee_ids[dst],
        #                     velo=self.args.ee_velo, from_root=self.args.ee_from_root)
        # height = self.models[dst].height[rnd_idx]
        # height = height.reshape((height.shape[0], 1, height.shape[1], 1))
        # fake_ee = fake_ee / height
        
        # self.fake_latent.append(fake_latent)
        # self.fake_pos.append(fake_pos)
        # self.fake_res.append(fake_res)
        # self.fake_ee.append(fake_ee)
        # self.fake_res_denorm.append(fake_res_denorm)    # [1,87,t]
        return fake_res_denorm  # 在测试环境下，就是1v1产生结果（实时性的要求）
        
    def retargetForward2(self, motion_input, offidx):
        self.offset_repr = []
        
        for i in range(self.n_topology):    # self.dataset.offsets[i]): [n,v,3]
            self.offset_repr.append(self.models[i].static_encoder(self.dataset.offsets[i]))  # [n,v,3] + [n,6v'] + [n,12v']

        # 重建过程，构造独立于骨骼的动作表示
        # motion, offset_idx = self.motions_input[0]  # [n,C(4v-4+3),t_w(64)],  一个int对应character的序号
        motion = motion_input.to(self.device)
        # self.motions.append(motion)                     # self.motions: 包含2个[n,C(4v-4+3),t_w(64)]的list 【<<1个】

        # motion_denorm = self.dataset.denorm(i, offset_idx, motion)  # 对应character的序号对motion去标准化
        # self.motion_denorm.append(motion_denorm)        # self.motion_denorm: 包含2个[n,C(4v-4+3),t_w(64)]的list 【<<2个】
        offsets = [self.offset_repr[0][p][offidx] for p in range(self.args.num_layers + 1)] # 提取该拓扑的offset
        latent, res = self.models[0].auto_encoder(motion, offsets)  # 编码后的动作[n, 4v''*4, t_w/4] + 再解码后的动作[n,4v-4+3, t_w]

        src = 0
        dst = 1
        rnd_idx = list(range(latent.shape[0]))
        dst_offsets_repr = [self.offset_repr[dst][p][rnd_idx] for p in range(self.args.num_layers + 1)]
        # 将编码后的动作self.latents，通过另一套骨骼offset解码，之后和重建流程类似，构建新模型新动作的pos、ee、height等数据
        # fake_res = self.models[dst].auto_encoder.dec(self.latents[src], dst_offsets_repr)
        fake_res = self.models[dst].auto_encoder.dec(latent, dst_offsets_repr)
        # fake_latent = self.models[dst].auto_encoder.enc(fake_res, dst_offsets_repr)

        fake_res_denorm = self.dataset.denorm(dst, rnd_idx, fake_res)
        return fake_res_denorm  # 在测试环境下，就是1v1产生结果（实时性的要求）
    
        
    def computeResult(self):
        r'''
            返回计算出的所有retarting结果
            在1类拓扑（只有0/1两类）有多个模型的情况下，返回[n,C(),t]
            一般来说只有一个模型，返回的就是[1,C(87),t]
        '''
        gt_poses = []
        gt_denorm = []
        # 真值，在实时计算和测试里不需要gt
        # for src in range(self.n_topology):
        #     gt = self.motion_backup[src]
        #     idx = list(range(gt.shape[0]))
        #     gt = self.dataset.denorm(src, idx, gt)
        #     gt_denorm.append(gt)
        #     gt_pose = self.models[src].fk.forward_from_raw(gt, self.dataset.offsets[src][idx])
        #     gt_poses.append(gt_pose)
        #     for i in idx:
        #         new_path = os.path.join(self.bvh_path, self.character_names[src][i])
        #         from option_parser import try_mkdir
        #         try_mkdir(new_path)
        #         self.writer[src][i].write_raw(gt[i, ...], 'quaternion', os.path.join(new_path, '{}_gt.bvh'.format(self.id_test)))
        src = 0
        dst = 1
        fake_denorms = []
        for i in range(len(self.character_names[dst])):
            # dst_path = os.path.join(self.bvh_path, self.character_names[dst][i])
            fake_denorms.append(self.fake_res_denorm[0][i, ...])
            # self.writer[dst][i].write_raw(self.fake_res_denorm[0][i, ...], 'quaternion',
            #                                 os.path.join(dst_path, '{}_{}.bvh'.format(self.id_test, src)))

        # self.id_test += 1
        return fake_denorms
        