import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib as mpl

# 设置论文级别的字体和样式
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'mathtext.fontset': 'stix',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})


def create_frontal_smpl_model():
    # 创建高分辨率图形 - 使用2D图
    fig, ax = plt.subplots(figsize=(8, 12), dpi=300)

    # 定义颜色方案
    colors = {
        'pelvis': '#4477AA',  # 蓝色
        'spine': '#66CCEE',  # 浅蓝色
        'head': '#228833',  # 绿色
        'left_arm': '#CCBB44',  # 黄色
        'right_arm': '#EE6677',  # 红色
        'left_leg': '#AA3377',  # 紫色
        'right_leg': '#BBBBBB',  # 灰色
        'joint': '#000000',  # 黑色
        'label_bg': '#FFFFFF',  # 白色
        'outline': '#555555'  # 深灰色
    }

    # 定义SMPL关节点位置 - 2D正面视图 (x, y)
    # 只使用x和z坐标，忽略y坐标（深度）
    joints = np.array([
        [0.0, 0.0],  # 0: Pelvis
        [-0.1, -0.1],  # 1: Left Hip
        [0.1, -0.1],  # 2: Right Hip
        [0.0, 0.1],  # 3: Spine1
        [-0.18, -0.3],  # 4: Left Knee
        [0.18, -0.3],  # 5: Right Knee
        [0.0, 0.2],  # 6: Spine2
        [-0.18, -0.5],  # 7: Left Ankle
        [0.18, -0.5],  # 8: Right Ankle
        [0.0, 0.3],  # 9: Spine3
        [-0.18, -0.6],  # 10: Left Foot
        [0.18, -0.6],  # 11: Right Foot
        [0.0, 0.45],  # 12: Neck
        [-0.05, 0.35],  # 13: Left Collar
        [0.05, 0.35],  # 14: Right Collar
        [0.0, 0.6],  # 15: Head
        [-0.2, 0.35],  # 16: Left Shoulder
        [0.2, 0.35],  # 17: Right Shoulder
        [-0.35, 0.25],  # 18: Left Elbow
        [0.35, 0.25],  # 19: Right Elbow
        [-0.45, 0.1],  # 20: Left Wrist
        [0.45, 0.1],  # 21: Right Wrist
        [-0.5, 0.0],  # 22: Left Hand
        [0.5, 0.0]  # 23: Right Hand
    ])

    # 定义骨骼连接
    bones = [
        # 躯干
        (0, 1), (0, 2), (0, 3),  # 骨盆到左/右髋关节和脊柱1
        (3, 6), (6, 9), (9, 12), (12, 15),  # 脊柱到颈部到头部
        (9, 13), (9, 14),  # 脊柱3到左/右锁骨
        (13, 16), (14, 17),  # 锁骨到肩膀

        # 左腿
        (1, 4), (4, 7), (7, 10),  # 左髋到左膝到左踝到左脚

        # 右腿
        (2, 5), (5, 8), (8, 11),  # 右髋到右膝到右踝到右脚

        # 左臂
        (16, 18), (18, 20), (20, 22),  # 左肩到左肘到左腕到左手

        # 右臂
        (17, 19), (19, 21), (21, 23)  # 右肩到右肘到右腕到右手
    ]

    # 为不同的骨骼部位指定颜色
    bone_colors = {}
    for bone in bones:
        if bone in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]:
            bone_colors[bone] = colors['spine']
        elif bone in [(9, 13), (13, 16), (16, 18), (18, 20), (20, 22)]:
            bone_colors[bone] = colors['left_arm']
        elif bone in [(9, 14), (14, 17), (17, 19), (19, 21), (21, 23)]:
            bone_colors[bone] = colors['right_arm']
        elif bone in [(0, 1), (1, 4), (4, 7), (7, 10)]:
            bone_colors[bone] = colors['left_leg']
        elif bone in [(0, 2), (2, 5), (5, 8), (8, 11)]:
            bone_colors[bone] = colors['right_leg']

    # 主要关节点和对应样式
    key_joints = {
        0: {'label': 'Pelvis', 'size': 400, 'color': colors['pelvis']},
        3: {'label': 'Spine1', 'size': 300, 'color': colors['spine']},
        9: {'label': 'Spine3', 'size': 300, 'color': colors['spine']},
        12: {'label': 'Neck', 'size': 300, 'color': colors['spine']},
        15: {'label': 'Head', 'size': 400, 'color': colors['head']}
    }

    # 先绘制人体轮廓（作为背景）
    # 头部
    head = Circle((joints[15][0], joints[15][1]), 0.12, fill=True, alpha=0.2, color=colors['head'])
    ax.add_patch(head)

    # 躯干
    torso_width = 0.3
    torso_height = joints[12][1] - joints[0][1]
    torso = Rectangle((joints[0][0] - torso_width / 2, joints[0][1]),
                      torso_width, torso_height, fill=True, alpha=0.2, color=colors['spine'])
    ax.add_patch(torso)

    # 绘制骨骼
    for bone in bones:
        p1, p2 = joints[bone[0]], joints[bone[1]]
        color = bone_colors.get(bone, colors['outline'])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, linewidth=4, alpha=0.8)

    # 绘制关节点
    for i, joint in enumerate(joints):
        if i in key_joints:
            style = key_joints[i]
            ax.scatter(joint[0], joint[1],
                       color=style['color'], s=style['size'],
                       edgecolor='black', linewidth=0.7, alpha=0.9, zorder=10)
        else:
            ax.scatter(joint[0], joint[1],
                       color='white', s=200,
                       edgecolor='black', linewidth=0.7, alpha=0.9, zorder=10)

    # 添加关节点编号
    for i, joint in enumerate(joints):
        ax.text(joint[0], joint[1], f"{i}",
                fontsize=9, fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor=colors['label_bg'], alpha=0.8,
                          edgecolor='black', boxstyle='round,pad=0.1', linewidth=0.5),
                zorder=15)

    # 为主要关节添加标签
    for i, style in key_joints.items():
        joint = joints[i]
        # 调整标签位置以避免重叠
        offset_x = -0.15 if i in [0, 3, 9, 12, 15] else 0.05
        offset_y = 0.03
        ax.text(joint[0] + offset_x, joint[1] + offset_y, style['label'],
                fontsize=9, ha='right', color='black',
                bbox=dict(facecolor=colors['label_bg'], alpha=0.8,
                          edgecolor=style['color'], boxstyle='round,pad=0.2', linewidth=1),
                zorder=15)

    # 添加左右指示
    ax.text(-0.45, -0.45, "LEFT", fontsize=12, fontweight='bold', color=colors['left_arm'])
    ax.text(0.45, -0.45, "RIGHT", fontsize=12, fontweight='bold', color=colors['right_arm'])

    # 设置坐标轴和标题
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_title('SMPL 24-Joint Human Body Model (Front View)', fontweight='bold', pad=20)

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors['spine'], lw=4, label='Spine/Torso'),
        Line2D([0], [0], color=colors['left_arm'], lw=4, label='Left Arm'),
        Line2D([0], [0], color=colors['right_arm'], lw=4, label='Right Arm'),
        Line2D([0], [0], color=colors['left_leg'], lw=4, label='Left Leg'),
        Line2D([0], [0], color=colors['right_leg'], lw=4, label='Right Leg')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9, edgecolor='black')

    # 设置相等的纵横比
    ax.set_aspect('equal')

    # 删除坐标轴和刻度
    ax.set_axis_off()

    # 添加节点说明表格
    table_data = [
        ["0", "Pelvis/Root"],
        ["1", "Left Hip"],
        ["2", "Right Hip"],
        ["3", "Spine1"],
        ["4", "Left Knee"],
        ["5", "Right Knee"],
        ["6", "Spine2"],
        ["7", "Left Ankle"],
        ["8", "Right Ankle"],
        ["9", "Spine3"],
        ["10", "Left Foot"],
        ["11", "Right Foot"],
        ["12", "Neck"],
        ["13", "Left Collar"],
        ["14", "Right Collar"],
        ["15", "Head"],
        ["16", "Left Shoulder"],
        ["17", "Right Shoulder"],
        ["18", "Left Elbow"],
        ["19", "Right Elbow"],
        ["20", "Left Wrist"],
        ["21", "Right Wrist"],
        ["22", "Left Hand"],
        ["23", "Right Hand"]
    ]

    # 将节点说明添加到右侧
    # 创建一个右边的子图用于显示节点说明
    plt.figtext(0.75, 0.5, 'SMPL Joint Definitions:', fontsize=11,
                fontweight='bold', ha='center', va='center')

    # 添加简单的表格文本形式
    col_width = 0.12
    col_height = 0.02
    for i, (idx, name) in enumerate(table_data):
        row = i % 12
        col = i // 12
        x_pos = 0.65 + col * col_width * 3
        y_pos = 0.7 - row * col_height * 2
        plt.figtext(x_pos, y_pos, f"{idx}: {name}", fontsize=8)

    plt.tight_layout()

    # 保存高分辨率图像
    plt.savefig('smpl_24_joints_frontal.png', dpi=300, bbox_inches='tight', format='png')
    plt.savefig('smpl_24_joints_frontal.pdf', dpi=300, bbox_inches='tight', format='pdf')

    plt.show()

    print("图像已保存为: smpl_24_joints_frontal.png, smpl_24_joints_frontal.pdf")


# 创建正面朝向的2D SMPL模型图
create_frontal_smpl_model()
