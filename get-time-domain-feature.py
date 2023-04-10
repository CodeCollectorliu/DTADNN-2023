import pywt
import numpy as np
import os
from matplotlib import pyplot as plt

import feature

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_path1 = r"\35Hz12kNxnpy"
# data_path1 = r"\Full_Test_set"
data_path1 = r'{0}\\'.format(data_path1)
data_names1 = list(os.listdir(data_path1))
# data_save_path = r"\data\Test"
data_save_path = r"\data\rawtd1"
if os.path.isdir(data_save_path) == False:  # 如果该文件不存在，就创建该文件
    os.mkdir(data_save_path)  # 前提是这个路径是存在的
data_save_path = r'{0}\\'.format(data_save_path)

for name in data_names1:
    loc = data_path1 + name
    data1 = np.load(loc)
    data1 = data1[:, :2560]
    data1_features = np.zeros((8, data1.shape[0]))
    for i in range(data1.shape[0]):
        data1_features[:, i] = feature.time_fea(data1[i, :])
    t = range(data1_features.shape[1])
    feature_name = [
        "1-RMS均方根", "2-能量", "3-熵", "4-最大绝对值", "5-平均绝对值", "6-标准差", "7-裕度因子",
        "8-asinh"
    ]
    np.save(data_save_path + name, data1_features)
    for i in range(data1_features.shape[0]):
        img_name1 = r'C:\Users\11484\Documents\pythonProject\xijiao-cwt-CNN2d-文章-注意力\时域分析'
        img_name1 = '{0}\\'.format(img_name1) + str(name[:-4])
        if os.path.isdir(img_name1) == False:  # 如果该文件不存在，就创建该文件
            os.mkdir(img_name1)  # 前提是这个路径是存在的
        img_name1 = '{0}\\'.format(img_name1)
        img_name = img_name1 + r'\第' + str(i + 1) + '个特征.png'
        plt.figure(figsize=(8, 3))
        plt.plot(t, data1_features[i, :])
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(feature_name[i])
        plt.savefig(img_name)
        plt.clf()
        plt.cla()
        plt.close()
