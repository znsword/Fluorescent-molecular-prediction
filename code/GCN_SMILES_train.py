import torch
import torch.nn as nn
import dgl.function as fn
from dgllife.model.readout import MLPNodeReadout
from dgllife.model.gnn.gcn import GCNLayer
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
import deepchem as dc
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import dgl
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.utils import shuffle

flag = 0  # 0为图神经模型，1为自注意力机制模型

config = {
    'train_epochs': 1500,
    'eval_epochs': 1,
    'batch_size': 128,
    'alpha': 1.0,
    'train_shuffle': True,
    'eval_shuffle': False,
    'device': 'cuda:0',
    'number_atom_features': 30,
    'number_bond_features': 11,
    'num_layers': 1,
    'num_timesteps': 1,
    'graph_feat_size': 16,
    'AFP_feat_size': 50,
    'dropout': 0.1,
    'n_classes': 2,
    'nfeat_name': 'x',
    'efeat_name': 'edge_attr',
    'g_lr': 1e-2,
    'a_lr': 1e-2,
    'filepath': 'database/SMILES-Lambda.csv'
}
'''
    train_epochs：训练的迭代次数。
    eval_epochs：模型评估的迭代次数。
    batch_size：每个批次的样本数量。
    alpha：图神经网络中的阻尼系数，用于控制信息传递的强度。
    train_shuffle：是否在训练过程中随机打乱数据。
    eval_shuffle：是否在评估过程中随机打乱数据。
    device：计算设备的名称（例如 cuda:0 表示使用第一个 GPU）。
    number_atom_features：分子节点（原子）的特征数量。
    number_bond_features：分子边（化学键）的特征数量。
    num_layers：图神经网络中的层数。
    num_timesteps：在每个时间步骤中将信息传递多少次。
    graph_feat_size：图级别的特征向量大小。
    dropout：在训练过程中使用的 dropout 概率。
    n_classes：分类任务的类别数量。
    nfeat_name：节点特征张量的名称。
    efeat_name：边特征张量的名称。
    learning_rate：优化器的学习率。
    filepath: 输入文件
'''


class mulGCN(torch.nn.Module):
    def __init__(self,  # 初始化
                 node_feat_size,  # 节点维度
                 edge_feat_size,  # 边维度
                 num_layers=1,  # GCNLayer层的数量
                 num_timesteps=1,  # MLPNodeReadout层的数量
                 graph_feat_size=10,  # 图特征维度
                 n_tasks=1,  # 输出的任务数量
                 dropout=0.):
        super(mulGCN, self).__init__()
        self.gnn = GCNLayer(in_feats=node_feat_size,  # 输入节点
                            out_feats=node_feat_size,  # 输出节点
                            gnn_norm='none',  # 归一化方式，这里不归一
                            activation=F.relu,  # 激活函数
                            residual=True,  # 残差链接
                            batchnorm=False,  # 批归一化，不归一
                            dropout=dropout)
        self.readout = MLPNodeReadout(node_feats=node_feat_size,  # 节点特征维度
                                      hidden_feats=node_feat_size,  # 隐藏层维度
                                      graph_feats=graph_feat_size,  # 图特征维度
                                      activation=F.relu,
                                      mode='sum')  # 节点特征的汇聚方式
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        node_feats = self.gnn(g, node_feats)
        g_feats = self.readout(g, node_feats)

        return self.predict(g_feats).view(-1, )


class mulAttentiveFP(torch.nn.Module):
    def __init__(self,  # 初始化
                 node_feat_size,  # 节点维度
                 edge_feat_size,  # 边维度
                 num_layers=1,  # GCNLayer层的数量
                 num_timesteps=1,  # MLPNodeReadout层的数量
                 graph_feat_size=200,  # 图特征维度
                 n_tasks=1,  # 输出的任务数量
                 dropout=0.):
        super(mulAttentiveFP, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,  # 输入节点
                                  edge_feat_size=edge_feat_size,  # 输出节点
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        node_feats = self.gnn(g, node_feats, edge_feats)
        g_feats = self.readout(g, node_feats)

        return self.predict(g_feats).view(-1, )


def get_batch(inputs, labels, st, batch_size):  # 批处理：列表，对应特征（这里是吸收波长），起始点，大小
    n = len(inputs)
    ed = min(n, st + batch_size)  # 批处理的结束位置，一般取st+batch_size，最后一组取n
    inputs = inputs[st:ed]

    g = dgl.batch(inputs)

    node = g.ndata['x']
    edge = g.edata['edge_attr']

    labels = torch.as_tensor(labels[st:ed])

    return g, node, edge, labels


def extract_feature(mol_smiles, mol_labels):
    # deepchem库分子图卷积特征提取器，将分子转化为具有固定长度的数值特征向量
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    mol_X = featurizer.featurize(mol_smiles)
    # 将提取的特征向量转化为图对象
    inputs = [
        graph.to_dgl_graph(self_loop=True).to(config['device']) for graph in mol_X
    ]

    assert len(inputs) == len(mol_labels)
    # 波长转换成张量对象
    mol_labels = torch.as_tensor(mol_labels, dtype=torch.float)

    return inputs, mol_labels


def mean_relative_error(y_pred, y_test):  # 求预测值与真实值的平均相对误差
    assert len(y_pred) == len(y_test)  # 确保训练输出与真实值长度一致
    mre = 0.0
    for i in range(len(y_pred)):
        mre = mre + abs((y_pred[i] - y_test[i]) / y_test[i])
    mre = mre * 100 / len(y_pred)
    return mre


# 计算平均绝对误差
def MAE(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    mae = 0
    for i in range(len(y_pred)):
        mae = mae + abs(y_pred[i] - y_test[i])
    mae = mae / len(y_pred)
    return mae


# 计算RMSE（均方根误差）
def RMSE(y_actual, y_pred):
    # 计算预测值与真实值之间的差异
    errors = y_pred - y_actual

    # 平方差值并求和
    squared_errors = np.square(errors)
    sum_squared_errors = np.sum(squared_errors)

    # 计算均方根误差
    rmse = np.sqrt(sum_squared_errors / len(y_actual))

    return rmse


if __name__ == '__main__':

    mol_df = pd.read_csv(config['filepath']).dropna()
    mol_df = shuffle(mol_df)
    train_df, valid_df = train_test_split(mol_df, test_size=0.3, random_state=1)
    train_df, test_df = train_test_split(mol_df, test_size=0.2, random_state=1)

    mol_smiles = train_df['molecule'].tolist()
    # mol_labels = train_df['lambda_abs'].tolist()
    mol_labels = train_df['lambda_em'].tolist()
    val_smiles = valid_df['molecule'].tolist()
    # val_labels = valid_df['lambda_abs'].tolist()
    val_labels = valid_df['lambda_em'].tolist()

    print('训练集：')
    for i, smiles in enumerate(mol_smiles):
        print(f"{i + 1}: {smiles}")

    inputs, mol_labels = extract_feature(mol_smiles, mol_labels)
    val_inputs, val_labels = extract_feature(val_smiles, val_labels)
    '''
    创建模型
    '''

    if flag == 0:
        model = mulGCN(
            node_feat_size=config['number_atom_features'],
            edge_feat_size=config['number_bond_features'],
            num_layers=config['num_layers'],
            num_timesteps=config['num_timesteps'],
            graph_feat_size=config['graph_feat_size'],
            n_tasks=1,
            dropout=config['dropout']
        )
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['g_lr'])
    elif flag == 1:
        model = mulAttentiveFP(
            node_feat_size=config['number_atom_features'],
            edge_feat_size=config['number_bond_features'],
            num_layers=config['num_layers'],
            num_timesteps=config['num_timesteps'],
            graph_feat_size=config['AFP_feat_size'],
            n_tasks=1,
            dropout=config['dropout'])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['a_lr'])

    '''
    数据预处理
    '''

    mol_labels, val_labels = torch.as_tensor(mol_labels).view(-1, 1), torch.as_tensor(val_labels).view(-1,
                                                                                                       1)  # 转化为2维的列向量
    # 节点特征和边特征的归一化或标准化处理可能会破坏它们之间的结构和关联性，从而导致模型无法充分利用图的结构信息。
    # 因此不对图的输入特征进行归一化或标准化
    min_max_scaler_y = MinMaxScaler()  # ！只能接受2维以上的输入
    min_max_scaler_y.fit(mol_labels)
    mol_labels, val_labels = torch.as_tensor(min_max_scaler_y.transform(mol_labels),
                                             dtype=torch.float), torch.as_tensor(min_max_scaler_y.transform(val_labels),
                                                                                 dtype=torch.float)  # 标准化，然后把输出从numpy转回张量
    mol_labels, val_labels = mol_labels.view(-1, ), val_labels.view(-1, )  # 转回一维行向量
    mol_labels, val_labels = mol_labels.to(config['device']), val_labels.to(config['device'])

    '''
    训练
    '''

    model.to(config['device'])
    model.train()  # 设置为训练模式
    lossF = nn.MSELoss()  # 计算损失
    losses = []  # 储存损失
    val_losses = []
    scheduler = StepLR(optimizer, step_size=750, gamma=0.1)  # 动态调整学习率

    for epoch in range(config['train_epochs']):
        for i in range(0, len(inputs), config['batch_size']):
            # 每轮开始清空梯度信息
            optimizer.zero_grad()
            # 获取批次信息
            g, node, edge, labels = get_batch(inputs, mol_labels, i, config['batch_size'])
            # 得到前向传播预测结果
            logits = model(g, node, edge, labels)
            # 获取验证集批次信息
            g, node, edge, val_labels = get_batch(val_inputs, val_labels, i, config['batch_size'])
            # 得到验证传播预测结果
            valids = model(g, node, edge, labels)
            # 计算损失，反向传播，更新模型参数
            loss = lossF(logits, labels)
            valids_loss = lossF(valids, val_labels)
            loss.backward()
            optimizer.step()
            # 将当前轮次损失添加进损失列表
            losses.append(loss.detach().cpu().item())
            val_losses.append(valids_loss.detach().cpu().item())
            # 输出训练信息
            print('epoch:{:d}/{:d} iter:{:d}/{:d} loss:{:.4f} lr={:.6f}'.format(epoch + 1, config['train_epochs'],
                                                                                (i + 1) // config['batch_size'],
                                                                                len(inputs) // config['batch_size'],
                                                                                sum(losses) / len(losses),
                                                                                scheduler.get_last_lr()[0]))

    '''
    测试
    '''

    # 设置为评估模式
    model.eval()
    logits, labels = [], []

    test_smiles = test_df['molecule'].tolist()
    test_labels = test_df['lambda_abs'].tolist()
    test_inputs, test_labels = extract_feature(test_smiles, test_labels)

    print('测试集：')
    for i, smiles in enumerate(test_smiles):
        print(f"{i + 1}: {smiles}")
    # 模型评估
    with torch.no_grad():
        for epoch in range(config['eval_epochs']):
            for ind, inp in enumerate(test_inputs):
                g = test_inputs[ind]
                label = test_labels[ind]
                node_feats = g.ndata[config['nfeat_name']]
                edge_feats = g.edata[config['efeat_name']]
                # 创建评估模型
                logit = model(g, node_feats, edge_feats)
                # 逆归一化
                logits_np = np.reshape(logit.cpu(), (-1, 1))
                logits_np = min_max_scaler_y.inverse_transform(logits_np).reshape(-1, )
                # 储存
                logits += list(logits_np)
                labels.append(label.cpu().tolist())

    out_df = pd.DataFrame({'labels': labels, 'preds': logits})
    out_df['abs_diff'] = out_df['labels'].sub(out_df['preds']).abs()
    # average_abs_diff = 100 * ((out_df['abs_diff'] / out_df['preds']).sum() / len(out_df))
    print('测试集输出结果：')
    print(out_df)
    y_test = np.array(labels)
    y_pred = np.array(logits)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    kendall_corr, _ = kendalltau(y_test, y_pred)
    mae = MAE(y_test, y_pred)
    rmse = RMSE(y_test, y_pred)
    mre = mean_relative_error(y_test, y_pred)
    Image_y_test, Image_y_predict, Image_Mre = [], [], []

    Image_y_test.append(y_test)
    Image_y_predict.append(y_pred)
    Image_Mre.append(mre)

    Image_y_test, Image_y_predict = np.reshape(Image_y_test, (-1,)), np.reshape(Image_y_predict, (-1,))  # 铺平
    print('MRE=', mre, 'MAE:', mae, 'RMSE:', rmse, 'pearson:', pearson_corr, 'spearman:', spearman_corr, 'kendall:',
          kendall_corr)

if mre <= 30:
    # # 根据 mre 的值构建文件名
    # save_folder = './发射/GCN/表格值/mulGCN'
    # save_filename = f'{mre}.txt'
    # save_path = f'{save_folder}/{save_filename}'
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # # 打开文件并将值写入
    # with open(save_path, 'w') as file:
    #     file.write(f'MRE={mre}\n')
    #     file.write(f'MAE={mae}\n')
    #     file.write(f'RMSE={rmse}\n')
    #     file.write(f'Pearson={pearson_corr}\n')
    #     file.write(f'Spearman={spearman_corr}\n')
    #     file.write(f'Kendall={kendall_corr}\n')
    #     file.write(f'y_test={Image_y_test}\n')
    #     file.write(f'y_predict={Image_y_predict}\n')

    # # 从第200个损失值开始，每隔100个样本取一个
    # train_loss_sampled = losses[200::100]
    # val_loss_sampled = val_losses[200::100]
    # # 计算采样点的 epochs
    # epochs = np.arange(200, len(train_loss_sampled) * 100 + 200, 100)
    # 将列表转换为NumPy数组
    losses_np = np.array(losses)

    # 找到第一个损失值小于或等于0.08的索引
    start_index = np.argmax(losses_np <= 0.08)

    # 从损失值小于或等于0.08的位置开始采样
    train_loss_sampled = losses[start_index::100]
    val_loss_sampled = val_losses[start_index::100]
    epochs = np.arange(start_index, len(train_loss_sampled) * 100 + start_index, 100)

    # 其他代码不变

    # 设置图像大小为 59cm x 59cm
    # 将mm转换为英寸
    mm_to_inch = 1 / 25.4

    # 设置图像大小为 65mm x 65mm
    fig_width_mm = 144
    fig_height_mm = 120

    fig_width_inch = fig_width_mm * mm_to_inch
    fig_height_inch = fig_height_mm * mm_to_inch
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=300)
    # 创建 FontProperties 对象并设置字体样式、大小和名称
    font = FontProperties()
    font.set_family('sans-serif')  # 无衬线字体样式
    font.set_name('Arial')  # 设置字体名称为 'Arial'
    font.set_size(12)  # 设置字体大小为 12
    errstr = 'AttFP'
    fig.text(0.4, 0.8, errstr, fontsize=12, weight='normal', ha='right', va='top', color='black')
    # 绘制损失曲线
    plt.plot(epochs, train_loss_sampled, 'b--', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_loss_sampled, 'r--', linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss', fontproperties=font)
    plt.xlabel('Epochs', fontproperties=font, labelpad=0)
    plt.ylabel('Loss', fontproperties=font, labelpad=0)
    plt.legend(fontsize=10)
    # 指定保存路径
    save_folder = './支持信息/GCN/发射'
    save_filename = f'{mre}.png'
    save_path = f'{save_folder}/{save_filename}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(save_path)
    plt.show()

'''
预测值-真实值 曲线
'''
# # 创建列表对象用于作图
#
# xmin, xmax = Image_y_test.min(), Image_y_test.max()  # 定坐标值
#
# y_test = Image_y_test
# result = Image_y_predict
# # 设置图像大小为 59cm x 59cm
# # 将mm转换为英寸
# mm_to_inch = 1 / 25.4
# # 设置图像大小为 65mm x 65mm
# fig_width_mm = 120
# fig_height_mm = 120
# fig_width_inch = fig_width_mm * mm_to_inch
# fig_height_inch = fig_height_mm * mm_to_inch
# dpi = 300
# fig = plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
# # 设置坐标轴刻度的方向
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# # 设置 x 和 y 轴的标签
# font = FontProperties()  # 创建一个字体属性对象的步骤。通过这个对象，您可以定义要在绘图中使用的字体样式、大小和名称
#
# # 设置字体样式为 'sans-serif'，即无衬线字体样式，使用 Arial
# font.set_family('sans-serif')
# font.set_name('Arial')  # 设置字体名称为 'Arial'
# font.set_size(12)  # 设置字体大小
# plt.xlabel('$\mathsf{\lambda}_{\mathsf{em (pre.)}}$', fontsize=12, labelpad=0)  # fontsize=20表示字体大小为20
# plt.ylabel('$\mathsf{\lambda}_{\mathsf{em (exp.)}}$', fontsize=12,
#            labelpad=0)  # labelpad 参数用于设置标签（如 x 轴标签和 y 轴标签）与图表的距离
#
# xmin = min(y_test)  # 找出最小值
# xmax = max(y_test)  # 找出最大值
# xmin = 5 * round(xmin / 5)  # 将xmin取整为以0和5结尾的数字
# xmax = 5 * round(xmax / 5)
# x = xmin - 20
# y = xmax + 20
# # 设置刻度的大小
# # 设置刻度间隔为 25 一个刻度
# plt.xticks(np.arange(xmin, xmax, 50), fontproperties=font, size=12, color='black')
# plt.yticks(np.arange(xmin, xmax, 50), fontproperties=font, size=12, color='black')
# # 绘制一个虚线作为参考线
# plt.plot([x, y], [x, y], ':', linewidth=1.5, color='red')
# # 假设您希望文本位于图像的左上角，距离左边和上边各 20 个像素
# errstr = 'AttentiveFP@MRE=%.2f%%' % mre
# fig.text(0.65, 0.8, errstr, fontsize=12, weight='normal', ha='right', va='top', color='black')
#
# # 绘制了一个散点图，其中横坐标是真实值 y_test，纵坐标分别是预测值 result 和真实值 y_test，c为颜色，marker为标记形式
# plt.scatter(y_test, result, c=(33 / 255, 158 / 255, 188 / 255), marker='o', s=10, label='Predicted')  # 预测值用绿色点
# plt.scatter(y_test, y_test, c=(251 / 255, 132 / 255, 2 / 255), marker='o', s=10, label='Real')  # 真实值用红色X
# plt.axis([x, y, x, y])
# # 获取当前的坐标轴对象
# ax = plt.gca()
# ax.set_facecolor('#FFFFFF')  # 图的底色
# ax.tick_params(top=True, right=True)  # 用于配置坐标轴刻度线和标签的参数的函数
# # 隐藏右边和上面的刻度
# ax.tick_params(right=False, top=False)
# # 调整图像的布局
# plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
# # 指定保存路径
# save_folder = './发射/GCN/预测图像/mulGCN'
# save_filename = f'{mre}.png'
# save_path = f'{save_folder}/{save_filename}'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# plt.savefig(save_path)
# # plt.show()
