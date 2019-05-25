import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from pathlib import Path
from a_lib import *
from scipy.interpolate import interp1d


def Moving_Mean_Filter(Mov_in, n=3):
    """
    A moving mean filter.

    Parameter:
        Mov_in(list/np.arr): Input data.

    Return:
        (list): Filter output.
    """
    Mov_in = list(Mov_in)
    # padding
    padding = [Mov_in[0]]*int(1+(n-1)/2) + Mov_in + [Mov_in[-1]]*int((n-1)/2)
    Mean_Value_Buff = padding[:n]

    Mov_Out = []
    for item in padding[n:]:
        Mean_Value_Buff = Mean_Value_Buff[1:]+[item]
        Mov_Out.append(sum(Mean_Value_Buff)/n)
    return Mov_Out


def Derivative_Filter(Der_in):
    """
    导数滤波器

    parameter:
        B_in(list): The output of smoothing()

    return:
        list: [[X of derivative result], [Y of derivative result]]
    """
    Der_out = []
    last_interval = Der_in[0]
    for item in Der_in:
        Der_out.append((item-last_interval)/last_interval)
        last_interval = item
    return Der_out


def Upsampling(Ups_x, Ups_y, frequency=6):
    """
    上采样

    parameters:
        B(list): The output of B()
        frequency(int): The sampling frequency apply on B

    return:
        list: [[X of sampling result], [Y of sampling result]]
    """
    fl = interp1d(Ups_x, Ups_y, kind='linear')
    x_after_samp = np.linspace(0,
                               max(Ups_x//100),
                               int(max(Ups_x)//100*frequency),
                               endpoint=False)
    y_after_samp = fl(x_after_samp)
    
    return [[int(round(x)) for x in x_after_samp], y_after_samp]


def Add_label(Add_in, annotation, frequency=6):
    """
    将采样结束后的数据分割，每个片段对应60s

    parameters:
        seg_in(list): The 2nd item in the output list of C()
        frequency(int): The sampling frequency apply on B

    return:
        list: [[C[1] in 1st min], [C[1] in 2nd min], ...]
    """
    Add_out = []
    max_len = len(Add_in)
    for idx, val in enumerate(annotation):
        temp = Add_in[idx*350: (idx+1)*360]
        temp.append(val)
        Add_out.append(temp)
    return Add_out


my_fmt = ''
for idx in range(0, 360):
    my_fmt += ' %.4f'
my_fmt += ' %d'

file_path = 

a = ['a'+str(x).zfill(2) for x in range(1,21)]
b = ['b'+str(x).zfill(2) for x in range(1,6)]
c = ['c'+str(x).zfill(2) for x in range(1,11)]
datalist = a + b + c
for DataID in datalist:
    ecg, sqrs125, wavedet_, annotation = get_ecg_sqrs125_wavedet(DataID, 5)
    QRS = wavedet_
    # QRS = sqrs125

    # RR间期
    RR_interval_arr = np.array(QRS[1:]) - np.array(QRS[:-1])

    # 均值滤波
    After_mov_filter_list = Moving_Mean_Filter(RR_interval_arr)

    # 导数滤波
    After_der_filter_list = Derivative_Filter(After_mov_filter_list)

    # 维度均一化
    After_ups_filter_list = Upsampling(QRS[1:], After_der_filter_list)

    # 贴标签
    After_add_filter_list = Add_label(After_ups_filter_list, annotation)

    # 保存
    np.savetxt(file_path, np.array(After_add_filter_list), fmt=my_fmt)
pass
