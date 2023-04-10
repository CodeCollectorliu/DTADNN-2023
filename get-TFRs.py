import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import psutil
import numpy as np
import pywt
import scipy.io as scio
from mpl_toolkits.mplot3d import Axes3D
import pathlib
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def bilinear_interpolate(src, dst_size):
    height_src, width_src, channel_src = src.shape  # (h, w, ch)
    height_dst, width_dst = dst_size  # (h, w)
    """中心对齐，投影目标图的横轴和纵轴到原图上"""
    ws_p = np.array([(i + 0.5) / width_dst * width_src - 0.5
                     for i in range(width_dst)],
                    dtype=np.float32)
    hs_p = np.array([(i + 0.5) / height_dst * height_src - 0.5
                     for i in range(height_dst)],
                    dtype=np.float32)
    ws_p = np.repeat(ws_p.reshape(1, width_dst), height_dst, axis=0)
    hs_p = np.repeat(hs_p.reshape(height_dst, 1), width_dst, axis=1)
    """找出每个投影点在原图的近邻点坐标"""
    ws_0 = np.clip(np.floor(ws_p), 0,
                   width_src - 2).astype(int)  # type: ignore
    hs_0 = np.clip(np.floor(hs_p), 0, height_src - 2).astype(int)
    ws_1 = ws_0 + 1
    hs_1 = hs_0 + 1
    """四个临近点的像素值"""
    f_00 = src[hs_0, ws_0, :].T
    f_01 = src[hs_0, ws_1, :].T
    f_10 = src[hs_1, ws_0, :].T
    f_11 = src[hs_1, ws_1, :].T
    """计算权重"""
    w_00 = ((hs_1 - hs_p) * (ws_1 - ws_p)).T
    w_01 = ((hs_1 - hs_p) * (ws_p - ws_0)).T
    w_10 = ((hs_p - hs_0) * (ws_1 - ws_p)).T
    w_11 = ((hs_p - hs_0) * (ws_p - ws_0)).T
    """计算目标像素值"""
    return (f_00 * w_00).T + (f_01 * w_01).T + (f_10 * w_10).T + (f_11 *
                                                                  w_11).T


def getfig(data, i):
    data = data[:]
    fc = pywt.central_frequency(wavename)  # 中心频率
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)

    coefs, _ = pywt.cwt(
        data, scales,
        wavename)
    coefs = np.abs(coefs)
    coefs = np.expand_dims(coefs, axis=2)
    fig = bilinear_interpolate(coefs, (30, 30))
    coefs = np.squeeze(coefs)
    fig = fig[:, :, 0]
    lll = image_save_path + name[:-4]
    pathlib.Path(lll).mkdir(parents=True,
                            exist_ok=True)  # 前提是这个路径是存在的
    lll = '{0}\\'.format(lll)
    lll = lll + str(i) + '.jpg'
    plt.switch_backend('agg')
    plt.figure(figsize=(8, 8))
    plt.contourf(abs(fig), 100, cmap='rainbow')
    plt.axis("off")
    plt.savefig(lll, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
    return fig**2


data_path1 = r"\35Hz12kNxnpy"
# 35Hz12kNxnpy/37.5Hz11kNxnpy
# data_path1 = r"\PHM 2012\PHM 2012 npy\Full_Test_set"
data_path1 = r'{0}\\'.format(data_path1)
data_names1 = list(os.listdir(data_path1))
data_save_path = r"C:\Users\11484\Documents\pythonProject\phm-cwt-CNN2d-会议-注意力\data\rawtfd1"
image_save_path = r"C:\Users\11484\Documents\pythonProject\phm-cwt-CNN2d-会议-注意力\时频域分析"
if os.path.isdir(data_save_path) == False:  # 如果该文件不存在，就创建该文件
    os.mkdir(data_save_path)  # 前提是这个路径是存在的
if os.path.isdir(image_save_path) == False:  # 如果该文件不存在，就创建该文件
    os.mkdir(image_save_path)  # 前提是这个路径是存在的
data_save_path = '{0}\\'.format(data_save_path)
image_save_path = '{0}\\'.format(image_save_path)
wavename = "morl"
totalscal = 256
for name in data_names1:
    loc = data_path1 + name
    train_data = np.load(loc)
    train_data = train_data[:, :2560]
    # print(train_data.shape)
    xx = Parallel(n_jobs=psutil.cpu_count())(
        delayed(getfig)(train_data[i, :], i)
        for i in range(train_data.shape[0]))
    # ce = np.array(fig[0])
    coe = []
    figg = []
    for i in range(len(xx)):
        coe.append(xx[i][0])
        figg.append(xx[i][1])
    coe = np.array(coe)
    all_tfd1 = np.array(figg)
    print(coe.shape)
    temp = np.zeros((coe.shape[0], 1, coe.shape[2]))
    coe = np.concatenate((coe, temp), axis=1)
    print(coe.shape)
    coe = coe.reshape(coe.shape[0], 16, -1, coe.shape[2])
    print(coe.shape)
    coe = coe.reshape(coe.shape[0], 16, -1)
    all_at = np.mean(coe, axis=2)
    all_at1 = all_at.transpose()
    print(all_at1.shape)
    np.save(f'{data_save_path}at{name}', all_at1)
    np.save(f'{data_save_path}tfd{name}', all_tfd1)
    t = range(all_at1.shape[1])
    for m in range(16):
        image_path = image_save_path + name[:-4]
        if os.path.isdir(image_path) == False:  # 如果该文件不存在，就创建该文件
            os.mkdir(image_path)  # 前提是这个路径是存在的
        image_path = '{0}\\'.format(image_path)
        img_name1 = f'{image_path}第{str(m)}个特征'
        plt.switch_backend('agg')
        plt.figure(figsize=(8, 3))
        plt.plot(t, all_at1[m, :], color='r', linewidth=1)
        plt.xlabel("Time")
        plt.savefig(img_name1)
        plt.clf()
        plt.cla()
        plt.close()
