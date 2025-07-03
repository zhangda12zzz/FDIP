import graphviz

# --- Font Configuration ---
# !! 重要: 请确保你的系统安装了以下字体，或者替换为你系统上可用的中文字体 !!
# 常见的选择: "SimHei", "SimSun", "Microsoft YaHei", "WenQuanYi Zen Hei"
# 如果在 Linux 上，确保 fc-list 能找到你指定的字体
CHINESE_FONT = "SimHei"  # 或者 "Microsoft YaHei" 等

# Create a new Digraph
dot = graphviz.Digraph('FDIP_Framework', comment='FDIP 人体姿态估计框架')
dot.attr(rankdir='TB', label='FDIP 框架：基于IMU的人体姿态估计', fontsize='20', labelloc='t', compound='true')

# --- Global Graph Attributes for an "Academic" Look & Chinese Font ---
dot.attr('graph', fontname=CHINESE_FONT, splines='ortho', concentrate='true', nodesep='0.6', ranksep='0.8')
dot.attr('node', fontname=CHINESE_FONT, style='rounded,filled', fontsize='10', shape='box')
dot.attr('edge', fontname=CHINESE_FONT, fontsize='9')

# --- Node Styles (can be overridden per node if needed) ---
io_style = {'fillcolor': 'lightgrey'}
main_module_style = {'fillcolor': 'skyblue'}
# core_module_style is not used directly, clusters have their own colors
sub_process_style = {'fillcolor': 'aliceblue'}
fusion_style = {'fillcolor': 'lightpink'}
point_style = {'shape': 'point', 'width': '0.01', 'height': '0.01'}  # Invisible helper nodes

# --- Input Node ---
dot.node('imu_data', '6个IMU传感器\n(左/右脚踝, 头部, 左/右手腕, 骨盆)\n(加速度与方向数据)', **io_style)

# --- Serial Sub-networks RL and RA ---
dot.node('rl', '叶关节位置回归 (RL)\n(推断叶关节相对位置)', **main_module_style)
dot.node('ra', '全关节位置回归 (RA)\n(推断全关节相对位置)', **main_module_style)

dot.edge('imu_data', 'rl', label='稀疏IMU测量数据')
dot.edge('rl', 'ra', label='叶关节位置 (RL输出)')

# --- Pose Regression (RP) stage, containing the Core Sub-network ---
with dot.subgraph(name='cluster_rp') as c_rp:
    c_rp.attr(label='姿态回归 (RP) 阶段', style='filled', color='lightsteelblue', peripheries='1', fontsize='12',
              fontname=CHINESE_FONT)

    # Conceptual input point for data coming from RA into the RP stage
    c_rp.node('rp_entry_from_ra', **point_style, xlabel='来自RA (全关节位置)')  # xlabel for points

    # MSFKE Module (Core Part 1)
    with c_rp.subgraph(name='cluster_msfke') as c_msfke:
        c_msfke.attr(label='多尺度频域运动学编码器 (MSFKE)', style='filled', color='palegreen', peripheries='1',
                     fontsize='10', fontname=CHINESE_FONT)
        c_msfke.node('msfke_input_imu', 'IMU时序信号', **sub_process_style)
        c_msfke.node('dilated_conv', '空洞卷积\n(IMU分解为多尺度频域表示)', **sub_process_style)
        c_msfke.node('se_modules', '2x 挤压-激励 (SE) 模块\n(自适应加权, 抑制冗余)', **sub_process_style)
        c_msfke.node('msfke_output', '紧凑的全局躯体与细节运动\n频率混合表示', **sub_process_style)

        c_msfke.edge('msfke_input_imu', 'dilated_conv')
        c_msfke.edge('dilated_conv', 'se_modules')
        c_msfke.edge('se_modules', 'msfke_output')

    # DSTFPE Module (Core Part 2)
    with c_rp.subgraph(name='cluster_dstfpe') as c_dstfpe:
        c_dstfpe.attr(label='双流时空融合姿态估计器 (DSTFPE)', style='filled', color='lightgoldenrodyellow',
                      peripheries='1', fontsize='10', fontname=CHINESE_FONT)

        c_dstfpe.node('dstfpe_entry_from_msfke', **point_style, xlabel='来自MSFKE')  # xlabel for points

        c_dstfpe.node('st_gcn', '时空图卷积网络 (ST-GCN)\n(骨骼拓扑长期运动演变)', **sub_process_style)
        c_dstfpe.node('bigru', '双向门控循环单元 (biGRU)\n(四肢瞬时动态特征)', **sub_process_style)

        c_dstfpe.node('fusion', '多头注意力机制\n(特征融合)', **fusion_style)
        c_dstfpe.node('linear_regression', '线性回归层\n(推断姿态参数)', **main_module_style)

        # Edges within DSTFPE
        c_dstfpe.edge('dstfpe_entry_from_msfke', 'st_gcn', label='分流特征')
        c_dstfpe.edge('dstfpe_entry_from_msfke', 'bigru', label='分流特征')  # Duplicate label is fine for two streams
        c_dstfpe.edge('st_gcn', 'fusion')
        c_dstfpe.edge('bigru', 'fusion')
        c_dstfpe.edge('fusion', 'linear_regression', label='融合特征')

    # Connect MSFKE output to DSTFPE's entry point
    # Using lhead/ltail to connect to cluster boundaries for better visual flow with ortho splines
    c_rp.edge('msfke_output', 'dstfpe_entry_from_msfke', lhead='cluster_dstfpe', ltail='cluster_msfke')

    # Connect the output of RA (represented by rp_entry_from_ra) to the Linear Regression layer
    c_rp.edge('rp_entry_from_ra', 'linear_regression', label='全关节位置 (来自RA)\n(用于最终姿态估计)', style='dashed',
              lhead='cluster_dstfpe')

# --- Connections between main components ---
# Connect overall IMU data to MSFKE's specific IMU input (inside RP cluster)
dot.edge('imu_data', 'msfke_input_imu', label='IMU时序信号\n(MSFKE使用)', lhead='cluster_msfke')

# Connect RA's output to the entry point of the RP stage/cluster
dot.edge('ra', 'rp_entry_from_ra', lhead='cluster_rp', label='全关节位置 (RA输出)')

# --- Output Node ---
dot.node('final_pose', '最终姿态\n(全局位姿变换矩阵)', **io_style)
# Connect the output of RP (Linear Regression) to the final pose
dot.edge('linear_regression', 'final_pose', ltail='cluster_rp', label='估计的最终姿态')

# --- Rendering ---
# To render the graph, you would typically save it to a file:
output_filename = 'fdip_framework_flowchart_chinese'
try:
    dot.render(output_filename, view=False, format='png')  # Set view=True to auto-open
    print(f"流程图已保存为: {output_filename}.png 和 {output_filename}")
except graphviz.backend.execute.ExecutableNotFound:
    print("错误: Graphviz 'dot' 可执行文件未找到。请确保Graphviz已安装并已添加到系统PATH。")
    print("对于中文支持，请确保指定的字体（如SimHei）已安装。")
    print("\n你可以复制下面的DOT源码到在线Graphviz渲染器进行预览：")
    print("========================= DOT Source =========================")
    print(dot.source)
    print("==============================================================")

# print(dot.source) # Optionally print the DOT source
