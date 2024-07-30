from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pandas as pd
import tensorflow as tf
from pylab import *
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

filepath = './data/abs.csv'


def scheduler(epoch, lr):
    if epoch > 0 and epoch % 500 == 0:
        return lr * 0.1
    else:
        return lr


def set_InandOut(data):  # 设置输入与输出，根据输入的数据集属性调整
    set_X = data[['HOMO_LUMO_Gap', 'Solvent', 'dft']]  # 从表格中提取输入
    # set_X = data.drop('abs', axis=1)
    # set_Y = data[['abs']]  # 从表格中提取输出
    set_Y = data[['abs']]  # 从表格中提取输出
    return set_X, set_Y


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


'''
数据清洗
'''
data = pd.read_csv(filepath, encoding='gb18030')
print(f'原始数据：{data.shape}')
data = data.dropna()
print(f'数据清洗：{data.shape}')
data = data.sample(frac=1).reset_index(drop=True)  # 打乱样本
print('打乱样本')

# 切割设置训练值与测试值
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

# 设置输出与输出，注意类型都需要是dataFrame

train_data_x_df, train_data_y_df = set_InandOut(train_data)
print(f'输入样本：{train_data_x_df.shape}')
print(f'真实值：{train_data_y_df.shape}')

test_data_x_df, test_data_y_df = set_InandOut(test_data)
print(f'测试样本：{test_data_x_df.shape}')
print(f'真实值：{test_data_y_df.shape}')
model_mlp = RandomForestRegressor(max_depth=10,
                                  max_features=3,  # 最大特征数
                                  min_samples_leaf=6,  # 叶子结点的最小样本数
                                  n_estimators=150,  # 决策树的数量
                                  verbose=1,  # 设置为1以启用详细输出
                                  random_state=42)  # 为了结果可重现)g
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)  # 使用scheduler控制学习率,verbose=1用于进度条反馈
'''
数据处理
'''
# 归一化
min_max_scaler_X = MinMaxScaler()  # 归一化
min_max_scaler_y = MinMaxScaler()

min_max_scaler_X.fit(train_data_x_df)  # 拟合
min_max_scaler_y.fit(train_data_y_df)

train_x_trans = min_max_scaler_X.transform(train_data_x_df)  # 在fit基础上标准化数据
train_y_trans = min_max_scaler_y.transform(train_data_y_df)
test_x_trans = min_max_scaler_X.transform(test_data_x_df)
test_y_trans = min_max_scaler_y.transform(test_data_y_df)
'''
训练模型
'''
# 训练
X_train, y_train, X_test, y_test = train_x_trans, train_y_trans, test_x_trans, test_y_trans
y_train_ravel = y_train.ravel()
history = model_mlp.fit(X_train, y_train_ravel)
# 测试集做测试
y_test_predict = model_mlp.predict(X_test)

# 反缩放预测输出
y_test = test_data_y_df.reset_index(drop=True)

y_test_predict = y_test_predict.reshape(-1, 1)
y_test_predict = min_max_scaler_y.inverse_transform(y_test_predict)
y_predict = pd.DataFrame(y_test_predict)

# print(f'y_test：{y_test.shape}')
# print(f'y_predict：{y_predict.shape}')

result = pd.concat([y_test, y_predict], axis=1)
result = result.rename(columns={0: 'Pred'})
result = result.reset_index(drop=True)

abs_diff = abs(result.iloc[:, 0] - result.iloc[:, 1])
result['abs_diff'] = abs_diff

print(result)
pearson_corr, _ = pearsonr(y_test.values.ravel(), y_predict.values.ravel())
spearman_corr, _ = spearmanr(y_test.values.ravel(), y_predict.values.ravel())
kendall_corr, _ = kendalltau(y_test.values.ravel(), y_predict.values.ravel())
mae = MAE(y_test.values.ravel(), y_predict.values.ravel())
rmse = RMSE(y_test.values.ravel(), y_predict.values.ravel())
mre = mean_relative_error(y_test.values.ravel(), y_predict.values.ravel())
Image_y_test, Image_y_predict, Image_Mre = [], [], []

Image_y_test.append(y_test)
Image_y_predict.append(y_predict)
Image_Mre.append(mre)
Image_y_test, Image_y_predict = np.reshape(Image_y_test, (-1,)), np.reshape(Image_y_predict, (-1,))  # 铺平
print('MRE=', mre, 'MAE:', mae, 'RMSE:', rmse, 'pearson:', pearson_corr, 'spearman:', spearman_corr, 'kendall:',
      kendall_corr)
# 创建列表对象用于作图

y_test = Image_y_test
result = Image_y_predict
# 设置图像大小为 59cm x 59cm
# 将mm转换为英寸
mm_to_inch = 1 / 25.4
# 设置图像大小为 65mm x 65mm
fig_width_mm = 120
fig_height_mm = 120
fig_width_inch = fig_width_mm * mm_to_inch
fig_height_inch = fig_height_mm * mm_to_inch
dpi = 300
fig = plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)
# 设置坐标轴刻度的方向
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# 设置 x 和 y 轴的标签
font = FontProperties()  # 创建一个字体属性对象的步骤。通过这个对象，您可以定义要在绘图中使用的字体样式、大小和名称

# 设置字体样式为 'sans-serif'，即无衬线字体样式，使用 Arial
font.set_family('sans-serif')
font.set_name('Arial')  # 设置字体名称为 'Arial'
font.set_size(12)  # 设置字体大小
plt.xlabel('$\mathsf{\lambda}_{\mathsf{abs (pre.)}}$', fontsize=12, labelpad=0)  # fontsize=20表示字体大小为20
plt.ylabel('$\mathsf{\lambda}_{\mathsf{abs (exp.)}}$', fontsize=12,
           labelpad=0)  # labelpad 参数用于设置标签（如 x 轴标签和 y 轴标签）与图表的距离

xmin = min(y_test)  # 找出最小值
xmax = max(y_test)  # 找出最大值
xmin = 5 * round(xmin / 5)  # 将xmin取整为以0和5结尾的数字
xmax = 5 * round(xmax / 5)
x = xmin - 20
y = xmax + 20
# 设置刻度的大小
# 设置刻度间隔为 25 一个刻度
plt.xticks(np.arange(xmin, xmax, 50), fontproperties=font, size=12, color='black')
plt.yticks(np.arange(xmin, xmax, 50), fontproperties=font, size=12, color='black')
# 绘制一个虚线作为参考线
plt.plot([x, y], [x, y], ':', linewidth=1.5, color='red')
# 假设您希望文本位于图像的左上角，距离左边和上边各 20 个像素
errstr = 'RFR@MRE=%.2f%%' % mre
fig.text(0.65, 0.8, errstr, fontsize=12, weight='normal', ha='right', va='top', color='black')

# 绘制了一个散点图，其中横坐标是真实值 y_test，纵坐标分别是预测值 result 和真实值 y_test，c为颜色，marker为标记形式
plt.scatter(y_test, result, c=(33 / 255, 158 / 255, 188 / 255), marker='o', s=10, label='Predicted')  # 预测值用绿色点
plt.scatter(y_test, y_test, c=(251 / 255, 132 / 255, 2 / 255), marker='o', s=10, label='Real')  # 真实值用红色X
plt.axis([x, y, x, y])
# 获取当前的坐标轴对象
ax = plt.gca()
ax.set_facecolor('#FFFFFF')  # 图的底色
ax.tick_params(top=True, right=True)  # 用于配置坐标轴刻度线和标签的参数的函数
# 隐藏右边和上面的刻度
ax.tick_params(right=False, top=False)
# 调整图像的布局
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
# # 指定保存路径
# save_folder = './pictures'
# save_filename = f'{mre}.png'
# save_path = f'{save_folder}/{save_filename}'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# plt.savefig(save_path)
plt.show()
