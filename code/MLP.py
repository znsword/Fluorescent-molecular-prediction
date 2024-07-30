# MLP
import csv
import os
from itertools import islice
import matplotlib.pyplot as plt
import torch
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from scipy.stats import pearsonr, spearmanr, kendalltau
import tensorflow as tf


#  将输入的位串转换为表示属性的向量
# bitstr: 表示位串的字符串（或数字），例如："11001"
#  函数通过循环遍历输入的bitstr，将每个字符转换为整数，并添加到attr_vec中
#  如果输入位串是："11001"，则函数将返回属性向量[1, 1, 0, 0, 1]
def bit2attr(bitstr) -> list:
    attr_vec = list()
    for i in range(len(bitstr)):
        attr_vec.append(int(bitstr[i]))
    return attr_vec


#  计算平均相对误差
def mean_relative_error(y_pred, y_test):
    #  通过assert语句确保y_pred和y_test的长度相同，即预测值和真实值的数量一致。
    #  如果y_pred为预测结果列表 [10, 12, 8, 9]，而y_test为真实标签列表 [9, 11, 7, 10]，
    #  则函数将计算每个位置的相对误差，分别为 [0.1111, 0.0909, 0.1429, 0.1000]。
    #  最后将这四个相对误差累积并求平均值得到 MRE 值，假设四舍五入到小数点后两位，则 MRE 值为 11.13
    assert len(y_pred) == len(y_test)
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


Large_MRE_points = pd.DataFrame()
Large_MRE_X = []
Large_MRE_y_test = []
Large_MRE_y_pred = []
Large_MRE = []

'''
1) 数据预处理
'''
filepath = './data/abs.csv'

data = pd.read_csv(filepath, encoding='gb18030')
print(data.shape)  # 数据的行数和列数
data = data.dropna()  # 删除data DataFrame 中包含缺失值的行

print(data.shape)
data = shuffle(data)  # 对数据进行随机重排，避免模型对数据的任何顺序相关性的依赖
train_data, test_data = train_test_split(data, test_size=0.2)
# 将data中的特征和标签分开
data_x_df = train_data[['HOMO_LUMO_Gap', 'Solvent', 'dft']]  # 只保存特定的列
data_y_df = train_data[['abs']]  # 只保留‘label'这一列

# 归一化
min_max_scaler_X = MinMaxScaler()
min_max_scaler_X.fit(data_x_df)  # 对特征数据data_x_df进行拟合
x_trans1 = min_max_scaler_X.transform(data_x_df)  # 对特征数据data_x_df进行转换，得到缩放后的特征数据x_trans1。

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(data_y_df)
y_trans1 = min_max_scaler_y.transform(data_y_df)
print('test data: ', test_data.shape)

test_data_x_df = test_data[['HOMO_LUMO_Gap', 'Solvent', 'dft']]
test_data_y_df = test_data[['abs']]
x_trans1_test = min_max_scaler_X.transform(test_data_x_df)
y_trans1_test = min_max_scaler_y.transform(test_data_y_df)
# 保存归一化参数
# dump(min_max_scaler_X, 'min_max_scaler_X.joblib')
# dump(min_max_scaler_y, 'min_max_scaler_y.joblib')
'''
3) 构建模型
'''

from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Dropout
from keras import models, regularizers
from keras.optimizers import Adam, RMSprop, SGD


# 吸收
def buildModel():
    model = models.Sequential()
    # Dense层（全连接层）和Dropout层（随机失活层）。
    l4 = Dense(512, activation='relu')  # 定义一个包含512个神经元的全连接层，并使用ReLU（修正线性单元）激活函数。

    l6 = Dense(320, activation='relu')
    l5 = Dropout(rate=0.2)  # 定义一个随机失活层，用于在训练过程中随机地将部分神经元置零，以防止过拟合。
    l7 = Dense(192, activation='relu')
    l1 = Dropout(rate=0.2)  # 定义一个随机失活层，用于在训练过程中随机地将部分神经元置零，以防止过拟合。
    l8 = Dense(1)

    layers = [l4, l5, l6, l7, l1, l8]
    for i in range(len(layers)):
        model.add(layers[i])

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])
    # 创建了一个多层感知器回归模型，并使用了指定的参数进行初始化。在实际使用中，可以使用model_mlp.fit()方法对模型进行训练。
    return model


# #  发射
#
# def buildModel():
#     model = models.Sequential()
#     # Dense层（全连接层）和Dropout层（随机失活层）。
#     l4 = Dense(224, activation='relu')  # 定义一个包含512个神经元的全连接层，并使用ReLU（修正线性单元）激活函数。
#     l5 = Dropout(rate=0.2)  # 定义一个随机失活层，用于在训练过程中随机地将部分神经元置零，以防止过拟合。
#     l6 = Dense(384, activation='relu')
#     l1 = Dropout(rate=0.2)  # 定义一个随机失活层，用于在训练过程中随机地将部分神经元置零，以防止过拟合。
#     l7 = Dense(32, activation='relu')
#     l8 = Dense(1)
#
#     layers = [l4, l6, l1, l7, l8]
#     for i in range(len(layers)):
#         model.add(layers[i])
#
#     adam = Adam(lr=0.00058)
#     model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])
#     # 创建了一个多层感知器回归模型，并使用了指定的参数进行初始化。在实际使用中，可以使用model_mlp.fit()方法对模型进行训练。
#     return model


# 学习率调度器函数
def scheduler(epoch, lr):
    # 当训练周期数是500的倍数时（即每训练500个周期），学习率会乘以0.1，相当于将学习率减小为原来的10%。其余情况下，学习率保持不变
    if epoch > 0 and epoch % 500 == 0:
        return lr * 0.1
    else:
        return lr


'''
4) 训练模型
'''
from sklearn import metrics

# 使用sklearn.metrics中的函数来计算模型的性能指标。
# n_split = 10
mlp_scores = []  # 存储MLP（多层感知器）模型在交叉验证过程中的得分
MAEs = []  # 存储每次交叉验证中MLP模型的平均绝对误差
out_MAEs = []  # 存储MLP模型在测试集上的平均绝对误差。
# 用来存储在交叉验证过程中，MLP模型在训练集和测试集上的真实值和预测值。
in_y_test = []
in_y_pred = []
out_y_test = []
out_y_pred = []

X_train = x_trans1
y_train = y_trans1
# 创建LearningRateScheduler回调函数，并将定义好的scheduler函数作为参数传递给它。verbose=1表示在每个epoch结束后打印学习率的变化情况
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
model_mlp = buildModel()
# 使用训练数据X_train和y_train来训练MLP模型，.fit() 方法是 Keras 模型的训练函数，用于训练模型的权重参数以使其适应给定的训练数据
model_mlp.fit(X_train, y_train, epochs=1500, verbose=1, callbacks=[callback])

# 外部验证，对测试集进行验证
X_test = x_trans1_test
# 通过 model_mlp.predict(x_trans1_test) 得到对测试集的预测结果 result
result = model_mlp.predict(x_trans1_test)

y_trans1_test = np.reshape(y_trans1_test, (-1, 1))
# 将归一化后的 y_trans1_test 还原回原始的标签。
y_test = min_max_scaler_y.inverse_transform(y_trans1_test)
result = result.reshape(-1, 1)
# 对预测结果 result 也进行逆变换，将其恢复成原始的目标标签
result = min_max_scaler_y.inverse_transform(result)
# 计算平均相对误差
# mae = mean_relative_error(y_test, result)
# out_MAEs.append(mae)

Large_MRE_X = []  # Type of X_test??
Large_MRE_y_test = []
Large_MRE_y_pred = []
Large_MRE = []
# 将 X_test 的形状重新调整为 (样本数量, 特征数量)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
X_test = min_max_scaler_X.inverse_transform(X_test)
# 计算预测值和真实值的平均相对误差
for idx in range(len(y_test)):
    Large_MRE.append(mean_relative_error([result[idx]], [y_test[idx]])[0])
# 真实值 y_test 和预测值 result 通过 np.reshape 函数调整为一维数组
Large_MRE_y_test = list(np.reshape(y_test, (-1,)))
Large_MRE_y_pred = list(np.reshape(result, (-1,)))

temp = pd.DataFrame(X_test)
temp = pd.concat(
    [temp, pd.DataFrame({'Real Value': Large_MRE_y_test}), pd.DataFrame({'Predicted Value': Large_MRE_y_pred}),
     pd.DataFrame({'MRE': Large_MRE})], axis=1)
# temp = temp.sort_values(by='MRE', ascending=False)
# 数据框 temp 中的数据保存到一个 CSV 文件中
# temp.to_csv('temp.csv', encoding='gb18030', index=False)

out_y_test.append(y_test)
out_y_pred.append(result)
# 模型评估系数
mre = mean_relative_error(y_test, result)
mae = MAE(y_test, result)
rmse = RMSE(y_test, result)
y_actual = y_test.flatten()
y_pred = result.flatten()
pearson_corr, _ = pearsonr(y_actual, y_pred)
spearman_corr, _ = spearmanr(y_actual, y_pred)
kendall_corr, _ = kendalltau(y_actual, y_pred)
length = y_test.shape[0]
# length = int(len)
# 指定模型保存的文件夹路径
# if mre < 2:
#     save_folder = './model'
#     mre = mre[0]
#     model_filename = f"model_mlp_mre_{mre:.2f}.h5"  # 根据 mre 的值生成模型文件
#     model_mlp.save(os.path.join(save_folder, model_filename))
a = [item[0] for item in y_test]
b = [item[0] for item in result]
print(a)
print(b)
for i in range(length):
    # print('第', i, '个:')
    print('real：', y_test[i][0], 'Prediction：', result[i][0], '差值=', abs(result[i][0] - y_test[i][0]))
print('MRE=', mre, 'MAE:', mae, 'RMSE:', rmse, 'pearson:', pearson_corr, 'spearman:', spearman_corr, 'kendall:',
      kendall_corr)

# # 画图
# # 导入必要的库
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
#
# # 创建一个新的图形
# fig = plt.figure(figsize=(8, 6))
# # 设置坐标轴刻度的方向
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# # 设置 x 和 y 轴的标签
# font = FontProperties()  # 创建一个字体属性对象的步骤。通过这个对象，您可以定义要在绘图中使用的字体样式、大小和名称
# font.set_family('serif')  # 设置字体样式为 'serif'，即衬线字体样式
# font.set_name('Times New Roman')  # 设置字体名称
# font.set_size(10)  # 设置字体大小
# plt.xlabel('Experimental values for lambda (mm)', fontsize=15)  # fontsize=20表示字体大小为20
# plt.ylabel('Predicted values for lambda (mm)', fontsize=15)
# # 设置刻度的大小
# plt.xticks(fontproperties=font, size=15, color='red')
# plt.yticks(fontproperties=font, size=15, color='red')
# xmin = y_test.min()  # 找出最小值
# xmax = y_test.max()  # 找出最大值
# x = xmin - 20
# y = xmax + 20
# # 绘制一个虚线作为参考线
# plt.plot([x, y], [x, y], ':', linewidth=1.5, color='red')
# # 创建 MRE（均方根误差）字符串，显示在图中
# errstr = 'MRE=%.2f%%' % mre
# x_position = xmax - 50  # x 坐标，略微偏左
# y_position = xmax - 20  # y 坐标，略微偏上
# plt.text(x_position, y_position, errstr, fontsize=20, weight='bold', ha='right', va='top', color='green')
# # 绘制了一个散点图，其中横坐标是真实值 y_test，纵坐标分别是预测值 result 和真实值 y_test，c为颜色，marker为标记形式
# plt.scatter(y_test, result, c=(33 / 255, 158 / 255, 188 / 255), marker='o', s=80, label='Predicted')  # 预测值用绿色点
# plt.scatter(y_test, y_test, c=(251 / 255, 132 / 255, 2 / 255), marker='o', s=80, label='Real')  # 真实值用红色X
# plt.axis([x, y, x, y])
# # 获取当前的坐标轴对象
# ax = plt.gca()
# ax.set_facecolor('#FFFFFF')  # 图的底色
# ax.tick_params(top=True, right=True)  # 用于配置坐标轴刻度线和标签的参数的函数
# # 添加颜色条
# # cbar = plt.colorbar()
# # cbar.ax.tick_params(labelsize=16)
# # 显示绘图
# # plt.savefig('1.eps', format='eps', dpi=300)
# plt.show()
#
# # 查看模型的所有层
# # model_mlp.summary()
