import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # debug专用
import torch
import time

from model import create_model
from dataset import create_dataset
import option_parser


def eval_prepare(args):
    character = []
    file_id = []
    character_names = []
    character_names.append('Smpl')#args.input_bvh.split('/')[-2])
    character_names.append('Aj')#args.target_bvh.split('/')[-2])
    if args.test_type == 'intra':
        if character_names[0].endswith('_m'):
            character = [['BigVegas', 'BigVegas'], character_names]
            file_id = [[0, 0], [args.input_bvh, args.input_bvh]]
            src_id = 1
        else:
            character = [character_names, ['Goblin_m', 'Goblin_m']]
            file_id = [[args.input_bvh, args.input_bvh], [0, 0]]
            src_id = 0
    elif args.test_type == 'cross':
        if character_names[0].endswith('_m'):
            character = [[character_names[1]], [character_names[0]]]
            file_id = [[0], [args.input_bvh]]
            src_id = 1
        else:
            character = [[character_names[0]], [character_names[1]]]
            file_id = [[args.input_bvh], [0]]
            # debug: manual setting
            # file_id = [
            #     ['datasets/PIPres/s_01/01.pkl.bvh',
            #     'datasets/PIPres/s_01/02.pkl.bvh',
            #     'datasets/PIPres/s_01/03.pkl.bvh',
            #     'datasets/PIPres/s_01/04.pkl.bvh'], 
            # [0,1,2,3]], 

            src_id = 0
    else:
        raise Exception('Unknown test type')
    return character, file_id, src_id


def recover_space(file):
    l = file.split('/')
    l[-1] = l[-1].replace('_', ' ')
    return '/'.join(l)


def main():
    parser = option_parser.get_parser()
    parser.add_argument('--input_bvh', type=str, default='dataset/DIPres/01.pkl.bvh')#, required=True)
    parser.add_argument('--target_bvh', type=str, default='dataset/DIPres/01.pkl.bvh')#, required=True)
    parser.add_argument('--test_type', type=str, default='cross')#, required=True)
    parser.add_argument('--output_filename', type=str, default='dataset/DIPres/01_retarget_rnn_cat.bvh')#, required=True)

    args = parser.parse_args()

    # argsparse can't take space character as part of the argument
    args.input_bvh = recover_space(args.input_bvh)
    args.target_bvh = recover_space(args.target_bvh)
    args.output_filename = recover_space(args.output_filename)

    character_names, file_id, src_id = eval_prepare(args)
    input_character_name = 'Smpl'#args.input_bvh.split('/')[-2]
    output_character_name = 'Aj'#args.target_bvh.split('/')[-2]
    output_filename = args.output_filename

    test_device = args.cuda_device
    eval_seq = args.eval_seq

    para_path = os.path.join(args.save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)

    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False
    args.rotation = 'quaternion'
    args.eval_seq = eval_seq

    std_paths = [['dataset/DIPres/01.pkl.bvh'],['dataset/Mixamo/std_bvhs/Aj.bvh']]

    dataset = create_dataset(args, character_names, std_paths)

    model = create_model(args, character_names, dataset, std_paths)
    model.load(epoch=10000)

    # input_motion = []
    # for i, character_group in enumerate(character_names):
    #     input_group = []
    #     for j in range(len(character_group)):
    #         # for k in range(len(file_id[i][j])): # debug line
    #             # new_motion = dataset.get_item(i, j, file_id[i][j][k])
    #         new_motion = dataset.get_item(i, j, file_id[i][j])
    #         new_motion.unsqueeze_(0)
    #         new_motion = (new_motion - dataset.mean[i][j]) / dataset.var[i][j]
    #         input_group.append(new_motion)
    #     input_group = torch.cat(input_group, dim=0)
    #     input_motion.append([input_group, list(range(len(character_group)))])

    # model.set_input(input_motion)   # 2个【运动数据[1,C,t]，模型编号】构成的list，这就是输入数据
    # output_motion = model.SMPLtest()    # [C,t]

    i = 0
    character_group = character_names[0]
    input_group = []
    for j in range(len(character_group)):
        new_motion = dataset.get_item(i, j, file_id[i][j])
        new_motion.unsqueeze_(0)
        new_motion = (new_motion - dataset.mean[i][j]) / dataset.var[i][j]
        input_group.append(new_motion)
    input_group = torch.cat(input_group, dim=0)
    # input_motion.append([input_group, list(range(len(character_group)))])
    # output_motion = model.SMPLtest(input_group, [0]) # input_group:[1,87,t]
    
    frame = input_group.shape[-1]
    win_size = 64
    output_motion = []
    for w in range(frame-win_size):
        # start_time = time.time()
        input_motion_w = input_group[:,:,w:w+win_size]
        output_motion_w = model.SMPLtest(input_motion_w, [0])
        output_motion_w = output_motion_w[:,:,-1:]
        output_motion.append(output_motion_w)
        # end_tiem = time.time()
        # print('training time', end_tiem - start_time)
    output_motion_final = torch.cat(output_motion, dim=-1).squeeze(0)    #[1,87,t']=>[87,t']

    model.writer[1][0].write_raw(output_motion_final, 'quaternion',
                                            output_filename)

    # 复制文件或目录
    # os.system('cp "{}/{}/0_{}.bvh" "./{}"'.format(model.bvh_path, output_character_name, src_id, output_filename))


if __name__ == '__main__':
    main()
