from graphviz import Digraph


def create_yolo11hysz_graph():
    # 初始化有向图
    dot = Digraph(comment='YOLOv11-HYSZ', format='svg',
                  graph_attr={'rankdir': 'TB', 'nodesep': '0.5', 'ranksep': '0.8'},
                  node_attr={'shape': 'box', 'style': 'filled', 'fontname': 'Helvetica'})

    # ==================== 主干网络（Backbone） ====================
    with dot.subgraph(name='cluster_backbone') as backbone:
        backbone.attr(label='Backbone (Dual-Branch)', color='blue', fontsize='14')

        # 输入层
        backbone.node('input', 'Input\n4 channels (RGB+IR)',
                      shape='ellipse', color='orange')

        # 可见光分支
        with backbone.subgraph(name='cluster_visible') as vis:
            vis.attr(label='Visible Branch', color='gray80')
            vis.node('silence_vis', 'SilenceChannel[0:3]\n→ RGB',
                     shape='parallelogram', color='lightblue')
            vis.edges([('input', 'silence_vis')])

            # 可见光分支层级
            vis_nodes = [
                ('conv1_vis', 'Conv\n64c, 3x3, s2'),
                ('conv2_vis', 'Conv\n128c, 3x3, s2'),
                ('c3k2_1_vis', 'C3k2\n256c, False, 0.25'),
                ('conv3_vis', 'Conv\n256c, 3x3, s2'),
                ('c3k2_2_vis', 'C3k2\n512c, False, 0.25'),
                ('conv4_vis', 'Conv\n512c, 3x3, s2'),
                ('c3k2_3_vis', 'C3k2\n512c, True'),
                ('conv5_vis', 'Conv\n1024c, 3x3, s2'),
                ('c3k2_4_vis', 'C3k2\n1024c, True')
            ]
            for name, label in vis_nodes:
                vis.node(name, label, color='lightblue')

            # 连接可见光分支
            vis.edges([
                ('silence_vis', 'conv1_vis'),
                ('conv1_vis', 'conv2_vis'),
                ('conv2_vis', 'c3k2_1_vis'),
                ('c3k2_1_vis', 'conv3_vis'),
                ('conv3_vis', 'c3k2_2_vis'),
                ('c3k2_2_vis', 'conv4_vis'),
                ('conv4_vis', 'c3k2_3_vis'),
                ('c3k2_3_vis', 'conv5_vis'),
                ('conv5_vis', 'c3k2_4_vis')
            ])

        # 红外分支
        with backbone.subgraph(name='cluster_infrared') as ir:
            ir.attr(label='Infrared Branch', color='gray80')
            ir.node('silence_ir', 'SilenceChannel[3:4]\n→ IR',
                    shape='parallelogram', color='pink')
            ir.edges([('input', 'silence_ir')])

            # 红外分支层级
            ir_nodes = [
                ('conv1_ir', 'Conv\n64c, 3x3, s2'),
                ('conv2_ir', 'Conv\n128c, 3x3, s2'),
                ('c3k2_1_ir', 'C3k2\n256c, False, 0.25'),
                ('conv3_ir', 'Conv\n256c, 3x3, s2'),
                ('c3k2_2_ir', 'C3k2\n512c, False, 0.25'),
                ('conv4_ir', 'Conv\n512c, 3x3, s2'),
                ('c3k2_3_ir', 'C3k2\n512c, True'),
                ('conv5_ir', 'Conv\n1024c, 3x3, s2'),
                ('c3k2_4_ir', 'C3k2\n1024c, True')
            ]
            for name, label in ir_nodes:
                ir.node(name, label, color='pink')

            # 连接红外分支
            ir.edges([
                ('silence_ir', 'conv1_ir'),
                ('conv1_ir', 'conv2_ir'),
                ('conv2_ir', 'c3k2_1_ir'),
                ('c3k2_1_ir', 'conv3_ir'),
                ('conv3_ir', 'c3k2_2_ir'),
                ('c3k2_2_ir', 'conv4_ir'),
                ('conv4_ir', 'c3k2_3_ir'),
                ('c3k2_3_ir', 'conv5_ir'),
                ('conv5_ir', 'c3k2_4_ir')
            ])

        # 跨模态融合
        backbone.node('concat_p3', 'Concat\nP3 (256+256→512c)', shape='circle', color='green')
        backbone.node('concat_p4', 'Concat\nP4 (512+512→1024c)', shape='circle', color='green')
        backbone.node('concat_p5', 'Concat\nP5 (1024+1024→2048c)', shape='circle', color='green')
        backbone.edges([
            ('c3k2_2_vis', 'concat_p3'),
            ('c3k2_2_ir', 'concat_p3'),
            ('c3k2_3_vis', 'concat_p4'),
            ('c3k2_3_ir', 'concat_p4'),
            ('c3k2_4_vis', 'concat_p5'),
            ('c3k2_4_ir', 'concat_p5')
        ])

    # ==================== 特征增强层 ====================
    with dot.subgraph(name='cluster_enhancement') as enhance:
        enhance.attr(label='Feature Enhancement', color='purple', fontsize='14')
        enhance.node('sppf', 'SPPF\n5x5 Pooling', shape='hexagon', color='yellow')
        enhance.node('c2psa', 'C2PSA\nChannel-Spatial Attention', shape='diamond', color='red')
        enhance.edges([
            ('concat_p5', 'sppf'),
            ('sppf', 'c2psa')
        ])

    # ==================== 检测头（Head） ====================
    with dot.subgraph(name='cluster_head') as head:
        head.attr(label='Detection Head', color='darkgreen', fontsize='14')

        # FPN路径
        head.node('upsample1', 'Upsample\n2x (nearest)', shape='triangle', color='gray')
        head.node('concat_p4_head', 'Concat P4', shape='circle', color='green')
        head.node('c3k2_p4', 'C3k2\n512c', color='lightblue')

        head.edges([
            ('c2psa', 'upsample1'),
            ('upsample1', 'concat_p4_head'),
            ('concat_p4', 'concat_p4_head'),
            ('concat_p4_head', 'c3k2_p4')
        ])

        # 检测输出
        head.node('detect', 'Detect\nP3/P4/P5 Outputs', shape='doubleoctagon', color='darkred')
        head.edges([
            ('concat_p3', 'detect'),
            ('c3k2_p4', 'detect'),
            ('c2psa', 'detect')
        ])

    # ==================== 渲染与保存 ====================
    dot.render(filename='yolov11hysz_architecture', cleanup=True, format='svg')
    return dot


# 生成并保存图表
create_yolo11hysz_graph()