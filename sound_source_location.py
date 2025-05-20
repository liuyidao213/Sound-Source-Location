import numpy as np
import torch
import hdf5storage
import argparse
from torch.utils.data import DataLoader
import sys
import time
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

sys.path.append('modelclass')
sys.path.append('funcs')

# 随机种子（确保程序在不同电脑可复现）
SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True


# 角度距离计算
def angular_distance_compute(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)


# 准确率计算
def ACCcompute(MAE, th):
    acc = sum([error <= th for error in MAE]) / len(MAE)
    return acc


# 自定义数据集类，用于加载特征、声源角度标签和事件类别标签
class MyDataloaderClass(Dataset):

    def __init__(self, X_data, label1, label2):
        self.x_data = X_data.astype(np.float32)     # 修改；将输入数据转换为np.float32类型
        self.label1 = label1                        # 声源角度标签 (DoA)
        self.label2 = label2                        # 事件类别标签

        self.len = X_data.shape[0]                  # 存储数据集的样本数量，即 X_data 的第一维大小

    def __getitem__(self, index):           # 该方法用于根据索引 index 获取数据集中的一个样本及其对应的两个标签
        return self.x_data[index], self.label1[index], self.label2[index]

    def __len__(self):
        return self.len


# 对张量x进行最小-最大归一化处理
def minmax_scaletensor(x, faxis):
    xmin = x.min(faxis, keepdim=True)[0]
    xmax = x.max(faxis, keepdim=True)[0]
    scale = np.where((xmax - xmin) == 0, 1, (xmax - xmin))      # 避免了除零错误
    data_out = (x - xmin) / scale
    xnorm = (x - xmin) / scale
    return xnorm


# 对输入的二维或多维数组 data_in 进行最小-最大归一化处理；data_in示例形状: (270, 27, 306)
def minmax_norm2d(data_in, faxis):
    dmin = data_in.min(axis=faxis, keepdims=True)
    dmax = data_in.max(axis=faxis, keepdims=True)
    scale = np.where((dmax - dmin) == 0, 1, (dmax - dmin))
    data_out = (data_in - dmin) / scale
    data_out[data_in == 0] = 0.0
    return data_out


# 计算事件预测结果的准确率
def ACCevent(evepred, evegt):
    label = np.argmax(evepred, axis=1)
    acc = sum(label == evegt) / label.shape[0]

    return acc


 # 计算平均绝对误差（Mean Absolute Error, MAE）和准确率
def MAEeval(Y_pred_t, Yte):
    # ------------ error evaluate   ----------
    erI1 = []    # 初始化一个空列表，用于存储误差
    DoA = []        # 初始化一个空列表，用于存储预测的方向角（Direction of Arrival, DoA）
    for i in range(Yte.shape[0]):   # 遍历每个时间步的样本
        hyp = Y_pred_t[i]   # 获取当前时间步的预测结果

        gt = Yte[i]      # 获取当前时间步的真实标签
        pred = np.argmax(hyp)  # 找到预测结果中概率最大的索引，作为预测的类别
        ang = angular_distance_compute(gt, pred)[0]     # 调用 angular_distance_compute 函数计算真实标签和预测标签之间的角度距离
        erI1.append(ang)  # 将计算得到的角度距离添加到误差列表中
        DoA.append(pred)  # 将预测的方向角添加到 DoA 列表

    MAE1 = sum(erI1) / len(erI1)    # 计算平均绝对误差（MAE）
    ACC1 = ACCcompute(erI1, 5)      # 调用 ACCcompute 函数计算准确率，假设误差阈值为 5
    # print("Testing MAE:%.8f \t ACC: %.8f " % (MAE1, ACC1))
    return MAE1, ACC1               # 返回计算得到的 MAE 和准确率


#######################  Model ################


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()    # 调用父类 nn.Module 的构造函数

        # 618 dim: (306+ 312)=(6*51+8*39): gbcphat + mfcc
        # input feature: bs*time*618
        self.time = 27                  # 输入特征的时间维度
        # 嵌入提取模块
        self.MLP3 = nn.Sequential(
            nn.Unflatten(1, (1, 618)),

            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Flatten(),
            nn.Linear(16 * 618, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Unflatten(1, (1, 1024)),

            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Flatten(),
            nn.Linear(16 * 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),

        )

        # 用于方向角预测，同样包含两个线性层，最后输出 360 维的结果，可能代表 360 个不同的方向角度。
        self.DoALayer = nn.Sequential(
            nn.Unflatten(1, (1, 1024)),
            # 插入一维卷积层
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            # 调整维度以适应后续线性层
            nn.Flatten(),
            nn.Linear(16 * 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.MaxPool1d(2),
            # nn.AdaptiveAvgPool1d(512),  # 区别大吗？

            # 调整维度以适应卷积操作
            nn.Unflatten(1, (1, 512)),
            # 插入一维卷积层
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # 调整维度以适应后续线性层
            nn.Flatten(),
            nn.Linear(16 * 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),


            nn.Linear(512, 360, bias=True),
        )
        # 用于事件分类预测，同样包含两个线性层，最后经过 Sigmoid 激活函数输出 10 维的结果，每个维度的值在 [0, 1] 区间，可表示事件分类的概率。
        self.eveLayer = nn.Sequential(

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 10, bias=True),
            nn.Sigmoid()  # Sigmoid 激活函数，将输出值映射到 [0, 1] 区间

        )
        # 使用 nn.MaxPool1d 对时间维度进行最大池化操作，池化核大小为 time（即 27）
        self.pool = nn.MaxPool1d(kernel_size=self.time)

    def forward(self, x):
        bs, t, dim = x.shape
        input = x.reshape(-1, x.shape[-1])

        x1 = self.MLP3(input)  # ([bs, 27, 618])
        x1 = x1.reshape(bs, t, -1)
        x2 = self.pool(x1.transpose(1, 2)).transpose(1, 2).squeeze()  # max pooling

        DoApred = self.DoALayer(x2)  # 角度定位
        evepred = self.eveLayer(x2)  # 事件分类

        return DoApred, evepred


################################################################
#######################     Main    ############################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xinyuan experiments')     # 创建一个 ArgumentParser 对象，用于解析命令行参数
    parser.add_argument('-gpuidx', metavar='gpuidx', type=int, default=0, help='gpu number')
    parser.add_argument('-epoch', metavar='epoch', type=int, default=50)  # Options: gcc, melgcc
    parser.add_argument('-batch', metavar='batch', type=int, default=8)              # 修改2 ** 5
    parser.add_argument('-lr', metavar='lr', type=float, default=0.001)
    parser.add_argument('-wts0', metavar='wts', type=float, default=1)
    parser.add_argument('-model', metavar='model', type=str, default='None')

    parser.add_argument('-input', metavar='input', type=str, default="small")  #

    args = parser.parse_args()      # 解析命令行参数

BATCH_SIZE = args.batch
print("experiments - xinyuan", flush=True)

device = torch.device("cuda:{}".format(args.gpuidx) if torch.cuda.is_available() else 'cpu')
args.device = device
print(device, flush=True)

criterion = torch.nn.MSELoss(reduction='mean')  # 均方误差损失函数 MSELoss，用于回归任务
criterion2 = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数 CrossEntropyLoss，用于分类任务
wts0 = args.wts0        # DOA权重
wts1 = 1 - wts0         # sound class权重
print(args, flush=True)
print('localization wts = ' + str(wts0) + '  event wts1 =' + str(wts1))


def training(epoch):
    model.train()       # 将模型设置为训练模式

    print("start training epoch " + str(epoch))

    for batch_idx, (data, DoAgt, evegt) in enumerate(train_loader, 0):  # 使用 enumerate 函数遍历 train_loader 中的每个批次数据

        inputs, DoAgt = data.type(torch.FloatTensor).to(device), DoAgt.type(torch.FloatTensor).to(
            device)  # DoA；将输入数据和标签转换为 torch.FloatTensor 类型
        evegt = evegt.type(torch.FloatTensor).to(device).squeeze()  # sound class；将输入数据和标签转换为 torch.FloatTensor 类型

        # start training -  
        DoApred, evepred = model.forward(inputs)  # 调用模型的 forward 方法进行前向传播，得到方向角预测结果 DoApred 和事件分类预测结果 evepred
        loss = criterion(DoApred.double(), DoAgt.double())  # 计算方向角的均方误差损失
        loss_event = criterion2(evepred.double(), evegt.long())  # 事件分类的交叉熵损失

        loss = wts0 * loss + wts1 * loss_event      # 将两个损失按照权重 wts0 和 wts1 进行加权求和得到最终损失 loss

        optimizer.zero_grad()           # 梯度清零
        loss.backward()                 # 反向传播
        optimizer.step()                # 更新
        # scheduler.step()              # 学习率调度器
        if (round(train_loader.__len__() / 5 / 100) * 100) > 0 and batch_idx % (
                round(train_loader.__len__() / 5 / 100) * 100) == 0:               # 打印训练损失信息
            print("training - epoch%d-batch%d: loss=%.3f" % (epoch, batch_idx, loss.data.item()), flush=True)

    torch.cuda.empty_cache()    # 在每个训练周期结束后，调用 torch.cuda.empty_cache() 方法清空 GPU 缓存，以释放不必要的内存


def testing(Xte, Yte, evegt):  # Xte: feature, Yte: binary flag
    model.eval()        # 设置模型为评估模式

    print('start testing' + '  ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    Y_pred_t = []               # 用于存储方向角的预测结果
    evepred = np.zeros((len(Xte), 10))      # 用于存储事件分类的预测结果

    for ist in range(0, len(Xte), BATCH_SIZE):
        ied = np.min([ist + BATCH_SIZE, len(Xte)])

        # 把当前批次的 NumPy 数组转换为 PyTorch 张量，同时将数据类型转换为 torch.FloatTensor，并迁移到指定设备（如 GPU）上
        inputs = torch.from_numpy(Xte[ist:ied]).type(torch.FloatTensor).to(device)

        # 把事件分类预测结果 eve 从 GPU 移到 CPU 上，去除梯度信息，再转换为 NumPy 数组，存储到 evepred 对应的位置
        DoApred, eve = model.forward(inputs)        # 前向传播
        evepred[ist:ied] = eve.cpu().detach().numpy()

        # 若使用的设备是 CPU，直接将方向角预测结果 DoApred 去除梯度信息并转换为 NumPy 数组，添加到 Y_pred_t 列表；若使用的是 GPU，则先将其移到 CPU 上，再进行相同操作
        if device.type == 'cpu':
            Y_pred_t.extend(DoApred.detach().numpy())  # in CPU
        else:
            Y_pred_t.extend(DoApred.cpu().detach().numpy())  # in CPU

    # ------------ error evaluate   ----------

    MAE, ACC = MAEeval(Y_pred_t, Yte.astype('float32'))  # 计算方向角预测的平均绝对误差（MAE）和准确率（ACC）
    ACC2 = ACCevent(evepred, evegt.squeeze().astype('int64'))   # 计算事件分类的准确率（ACC2）

    # event classification evaluation
    torch.cuda.empty_cache()       # 清空 GPU 缓存
    return MAE, ACC, ACC2


# ############################# load the data and the model ##############################################################
modelname = args.model  # 从命令行参数 args 中获取模型名称
lossname = 'MSE'        # 设置损失函数名称为均方误差（MSE）
print(args, flush=True)

if args.input == "small":                           # 数据加载
    data = hdf5storage.loadmat('feat618dim.mat')    # 使用 hdf5storage.loadmat 函数加载 MATLAB 格式的数据文件
    print("use small debug set", flush=True)
else:
    data = hdf5storage.loadmat('featall.mat')
    print("use all set", flush=True)

# 数据预处理（对加载的数据进行随机打乱）
L = len(data['class'])
ridx = random.sample(range(0, L), L)  # 生成随机索引 ridx

event = data['class'][ridx, :] - 1  # 从打乱后的数据中提取事件标签 event
doa = data['doa'][ridx, :]  # 从打乱后的数据中提取方向角标签 doa
doafeat360 = data['doafeat360'][ridx, :]  # 方向角后验概率 doafeat360
feat = data['feat'][ridx, :]  # 特征 feat

# 特征归一化（将特征 feat 拆分为 feat1（GCCPHAT 特征）和 feat2（MFCC 特征），并对它们分别进行最小 - 最大归一化处理，然后再合并）
feat1, feat2 = feat[:, :, :306].astype(np.float16), feat[:, :, 306:].astype(np.float16)   # gccphat, mfcc       ##修改
feat1, feat2 = minmax_norm2d(feat1, faxis=2), minmax_norm2d(feat2, faxis=2)
# 注意：gccphat 功能不在频道顺序中
feat = np.concatenate((feat1, feat2), axis=2)

# 数据集划分（根据 70% 的比例划分训练集和测试集，分别得到事件标签、方向角标签、方向角后验概率和特征的训练集和测试集）
ratio = round(0.7 * L)
eventtr, eventte = event[:ratio, :], event[ratio:, :]
doatr, doate = doa[:ratio, :], doa[ratio:, :]
doafeat360tr, doafeat360te = doafeat360[:ratio, :], doafeat360[ratio:, :]
feattr, featte = feat[:ratio, :], feat[ratio:, :]

# 创建数据加载器
train_loader_obj = MyDataloaderClass(feattr, doafeat360tr, eventtr)
train_loader = DataLoader(dataset=train_loader_obj, batch_size=args.batch, shuffle=True, num_workers=0)

# 模型和优化器初始化
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)    # 使用 Adam 优化器初始化 optimizer，设置学习率和要优化的参数

print(model, flush=True)

######## Training + Testing #######

EP = args.epoch
MAE, ACC, ACC2 = np.zeros(EP), np.zeros(EP), np.zeros(EP)
# MAE、ACC 和 ACC2 分别是长度为 EP 的零数组，用于存储每一轮测试的平均绝对误差、方向角预测准确率和事件分类准确率

for ep in range(EP):
    training(ep)

    MAE[ep], ACC[ep], ACC2[ep] = testing(featte, doate, eventte)  # 调用 testing 函数进行测试
    print("Testing ep%2d:    MAE:%.2f \t ACC: %.2f | ACC2: %.2f " % (ep, MAE[ep], ACC[ep] * 100, ACC2[ep] * 100),
          flush=True)

print("finish all!", flush=True)

ACC_result_array = np.column_stack((
    np.arange(EP),  # epoch 保持整数
    np.round(ACC*100, 2),  # ACC 保留两位小数
))

MAE_result_array = np.column_stack((
    np.arange(EP),  # epoch 保持整数
    np.round(MAE, 2),  # ACC 保留两位小数
))

ACC2_result_array = np.column_stack((
    np.arange(EP),  # epoch 保持整数
    np.round(ACC2*100, 2)  # ACC2 保留两位小数
))

print("ACC Result array:\n", ACC_result_array)
print("MAE_result_array:\n", MAE_result_array)
print("ACC2 Result array:\n", ACC2_result_array)

# --------------- 核心：添加中文字体配置 ---------------
plt.rcParams.update({
    # 中文字体：使用系统自带的"黑体"（SimHei）
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei'],  # 优先使用黑体
    # 解决负号显示异常
    'axes.unicode_minus': False,
    # 全局字体大小（可选）
    'font.size': 12
})
# ------------------------------------------------------
max_epoch = int(EP)  # 获取最大Epoch值

# 生成2的倍数刻度（从0开始，步长2）
even_epochs = np.arange(0, max_epoch + 1, 5)  # 包含终点，确保最大Epoch若为偶数也能显示

# 创建画布
plt.figure(figsize=(10, 6))  # 宽高比

# 绘制ACC折线（蓝色，带圆形标记）
plt.plot(ACC_result_array[:, 0], ACC_result_array[:, 1], 'bo-', label='ACC (DOA预测准确率)', linewidth=2, markersize=8)

# 绘制ACC2折线（橙色，带方形标记）
# plt.plot(ACC2_result_array[:, 0], ACC2_result_array[:, 1], 'rs--', label='ACC2 (事件分类准确率)', linewidth=2, markersize=8)

# 美化图表
plt.title('DOAE准确率随Epoch变化', fontsize=18, pad=20)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('准确率(%)', fontsize=16)
plt.xticks(even_epochs)  # 强制显示所有epoch刻度
plt.grid(True, linestyle='--', alpha=0.7)  # 网格线
plt.ylim(0.00, 100.00)  # 限制y轴范围
plt.legend(fontsize=14)  # 显示图例
plt.tight_layout()  # 自动调整布局

# 保存图片（可选）
plt.savefig('training_acc.png', dpi=300, bbox_inches='tight')


# 创建画布
plt.figure(figsize=(10, 6))  # 宽高比

# 绘制MAE折线（红色，带方形标记）
plt.plot(MAE_result_array[:, 0], MAE_result_array[:, 1], 'rs-', label='MAE (DOA预测的平均绝对误差)', linewidth=2, markersize=8)

# 绘制ACC2折线（橙色，带方形标记）
# plt.plot(ACC2_result_array[:, 0], ACC2_result_array[:, 1], 'rs--', label='ACC2 (事件分类准确率)', linewidth=2, markersize=8)

# 美化图表
plt.title('MAE随Epoch变化', fontsize=18, pad=20)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('MAE(°)', fontsize=16)
plt.xticks(even_epochs)  # 强制显示所有epoch刻度
plt.grid(True, linestyle='--', alpha=0.7)  # 网格线
plt.ylim(0.00, 50.00)  # 限制y轴范围
plt.legend(fontsize=14)  # 显示图例
plt.tight_layout()  # 自动调整布局

# 保存图片（可选）
plt.savefig('training_acc.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
