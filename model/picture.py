import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib as mpl


def create_fdip_matplotlib():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'SimSun']  # 优先使用的中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(14, 10))

    # 定义模块位置和大小
    modules = {
        'input': {'x': 0.5, 'y': 5, 'width': 2, 'height': 1, 'label': 'IMU传感器数据\n(6个位置)'},
        'rl': {'x': 3, 'y': 5, 'width': 2, 'height': 1, 'label': 'RL子网络\n叶关节位置回归'},
        'conv': {'x': 6, 'y': 5, 'width': 2, 'height': 1, 'label': '多尺度空洞卷积'},
        'se_trunk': {'x': 6, 'y': 7, 'width': 2, 'height': 1, 'label': 'SE模块\n(躯干路径)'},
        'se_limb': {'x': 6, 'y': 3, 'width': 2, 'height': 1, 'label': 'SE模块\n(四肢路径)'},
        'ra': {'x': 3, 'y': 3, 'width': 2, 'height': 1, 'label': 'RA子网络\n全关节位置回归'},
        'stgcn': {'x': 9, 'y': 7, 'width': 2, 'height': 1, 'label': 'ST-GCN\n骨骼拓扑长期运动建模'},
        'bigru': {'x': 9, 'y': 3, 'width': 2, 'height': 1, 'label': 'biGRU\n四肢瞬时动态特征捕捉'},
        'attention': {'x': 12, 'y': 5, 'width': 2, 'height': 1, 'label': '多头注意力机制\n特征融合与增强'},
        'rp': {'x': 15, 'y': 5, 'width': 2, 'height': 1, 'label': 'RP子网络\n最终姿态参数估计'},
        'output': {'x': 18, 'y': 5, 'width': 2, 'height': 1, 'label': '姿态参数\n(关节位置和旋转)'}
    }

    # 绘制模块
    colors = {
        'input': 'lightblue',
        'rl': 'lightcyan',
        'conv': 'lightyellow',
        'se_trunk': 'lightyellow',
        'se_limb': 'lightyellow',
        'ra': 'lightcyan',
        'stgcn': 'lightpink',
        'bigru': 'lightpink',
        'attention': 'lightgreen',
        'rp': 'lightcyan',
        'output': 'lightblue'
    }

    for name, props in modules.items():
        rect = patches.Rectangle(
            (props['x'], props['y']),
            props['width'],
            props['height'],
            linewidth=1,
            edgecolor='black',
            facecolor=colors[name],
            alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(
            props['x'] + props['width'] / 2,
            props['y'] + props['height'] / 2,
            props['label'],
            ha='center',
            va='center',
            fontsize=9
        )

    # 绘制连接
    connections = [
        ('input', 'rl'),
        ('rl', 'conv'),
        ('conv', 'se_trunk'),
        ('conv', 'se_limb'),
        ('se_trunk', 'stgcn'),
        ('se_limb', 'bigru'),
        ('input', 'ra'),
        ('ra', 'bigru'),
        ('stgcn', 'attention'),
        ('bigru', 'attention'),
        ('attention', 'rp'),
        ('rp', 'output'),
        # 直接连接（虚线）
        ('stgcn', 'rp', 'dashed'),
        ('bigru', 'rp', 'dashed'),
    ]

    for conn in connections:
        start = modules[conn[0]]
        end = modules[conn[1]]
        style = 'solid' if len(conn) < 3 else conn[2]

        # 计算起点和终点
        if start['x'] < end['x']:  # 从左到右
            start_x = start['x'] + start['width']
            start_y = start['y'] + start['height'] / 2
            end_x = end['x']
            end_y = end['y'] + end['height'] / 2
        elif start['x'] > end['x']:  # 从右到左
            start_x = start['x']
            start_y = start['y'] + start['height'] / 2
            end_x = end['x'] + end['width']
            end_y = end['y'] + end['height'] / 2
        elif start['y'] < end['y']:  # 从下到上
            start_x = start['x'] + start['width'] / 2
            start_y = start['y'] + start['height']
            end_x = end['x'] + end['width'] / 2
            end_y = end['y']
        else:  # 从上到下
            start_x = start['x'] + start['width'] / 2
            start_y = start['y']
            end_x = end['x'] + end['width'] / 2
            end_y = end['y'] + end['height']

        ax.annotate('',
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(
                        arrowstyle='->',
                        linestyle=style,
                        color='black'
                    ))

    # 添加框图标题和标签
    ax.text(10, 9.5, 'FDIP: 基于多频段运动学建模的IMU人体姿态估计方法',
            ha='center', fontsize=16, fontweight='bold')

    # 设置大型模块边界（可选）
    # MSFKE 边界
    msfke = patches.Rectangle((5.5, 2.5), 3, 6, linewidth=2, edgecolor='blue',
                              fill=False, linestyle='--')
    ax.add_patch(msfke)
    ax.text(7, 8.8, '多尺度频域运动学编码器 (MSFKE)', ha='center', fontsize=10)

    # DSTFPE 边界
    dstfpe = patches.Rectangle((8.5, 2.5), 6, 6, linewidth=2, edgecolor='red',
                               fill=False, linestyle='--')
    ax.add_patch(dstfpe)
    ax.text(11.5, 8.8, '双流时空融合姿态估计器 (DSTFPE)', ha='center', fontsize=10)

    # 设置坐标轴
    ax.set_xlim(0, 20)
    ax.set_ylim(1, 10)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('fdip_architecture_matplotlib.png', dpi=300, bbox_inches='tight')
    plt.show()


# 测试可用字体（用于调试）
def list_available_fonts():
    import matplotlib.font_manager as fm
    font_names = sorted([f.name for f in fm.fontManager.ttflist])
    print("可用字体列表:")
    for font in font_names:
        print(font)


# 如果中文仍然显示为方框，可以取消下面这行的注释来检查系统可用字体
# list_available_fonts()

create_fdip_matplotlib()
