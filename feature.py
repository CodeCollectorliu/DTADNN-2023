import numpy as np
import pandas as pd
import scipy.stats


def getEntropy(s):
    # 找到各个不同取值出现的次数
    if not isinstance(s, pd.core.series.Series):
        s = pd.Series(s)
    prt_ary = s.groupby(by=s).count().values / len(s)
    return -(np.log2(prt_ary) * prt_ary).sum()


def time_fea(signal_):
    N: int = len(signal_)
    y = signal_
    t_mean_1 = np.mean(y)  # 1_均值（平均幅值）

    t_rms_2 = np.sqrt((np.mean(y**2)))  # 2_RMS均方根

    t_fgf_3 = (np.mean(np.sqrt(np.abs(y))))**2  # 3_方根幅值

    t_std_4 = np.std(y, ddof=1)  # 4_标准差

    t_skew_5 = scipy.stats.skew(y)  # 5_偏度 skewness

    t_kur_6 = scipy.stats.kurtosis(y)  # 6_峭度 Kurtosis

    t_shape_7 = (N * t_rms_2) / (np.sum(np.abs(y)))  # 7_波形因子 Shape fator

    t_margin_8 = np.max(y) / np.abs(
        np.sum(np.sqrt(np.abs(y - t_mean_1))) / N)  # 8_Margin index

    t_clear_9 = np.max(np.abs(y)) / t_fgf_3  # 9_裕度因子  Clearance Factor

    t_cres_10 = np.max(np.abs(y)) / t_rms_2  # 10_峰值因子 Crest Factor

    t_entropy_11 = getEntropy(y)  # 11_熵
    # print(t_entropy_11)

    t_energy_12 = np.mean(y**2)  # 12_能量

    t_max_abs_13 = np.max(np.abs(y))  # 13_最大绝对值

    t_mean_abs_14 = np.mean(np.abs(y))  # 14_平均绝对值

    # t_atanh_15 = np.std(np.log((1 + y) / (1 - y)) / 2, ddof=1)  # 15_atanh

    t_asinh_16 = np.std(np.log(y + np.sqrt(y**2 + 1)), ddof=1)  # 16_asinh

    return np.array([
        t_rms_2, t_energy_12, t_entropy_11, t_max_abs_13, t_mean_abs_14,
        t_std_4, t_clear_9, t_asinh_16
    ])
