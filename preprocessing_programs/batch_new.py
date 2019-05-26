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
    Ups_x = [0] + list(Ups_x)
    Ups_y = [0] + Ups_y
    fl = interp1d(Ups_x, Ups_y, kind='linear')
    x_after_samp = np.linspace(0,
                               max(Ups_x)//100*100,
                               int(max(Ups_x)//100*frequency),
                               endpoint=False)
    y_after_samp = fl(x_after_samp)
    
    return list(y_after_samp)


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
    for idx, val in enumerate(annotation):
        temp = Add_in[idx*360: (idx+1)*360]
        temp.append(val)
        Add_out.append(temp)
    return Add_out


half_sample_size = 30
my_fmt = ''
for idx in range(0, half_sample_size*2):
    my_fmt += '%.4f '
my_fmt += '%d'
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

    # # 均值滤波
    After_mov_filter_list = Moving_Mean_Filter(RR_interval_arr, 7)

    # # 导数滤波
    After_der_filter_list = Derivative_Filter(After_mov_filter_list)

    # # 维度均一化
    # After_ups_filter_list = Upsampling(QRS[1:], After_der_filter_list)

    # # 贴标签
    # After_add_filter_list = Add_label(After_ups_filter_list, annotation)

    file_path = '/Users/rex/python/Thesis_Open_Source/RR_left_n1_wavedet/' + DataID + '.txt'
    path = Path(file_path)
    if path.exists():
        print('update ', end="")
        path.unlink()  # delete file
    print(DataID)
    # Short_Label_Sign = False 

    # 维度均一化并贴标签
    T  = 6000
    QRS_list = list(QRS[1:])
    QL_len = len(QRS_list)
    wedge = 0
    sample_with_label = []
    
    for idx, label in enumerate(annotation):
        middle_Time = int(idx * T + T/2)
        for i in range(wedge, QL_len):
            if QRS_list[i] > middle_Time:
                middle_QRS_idx = wedge = i
                break
        sample_list = list(After_der_filter_list[middle_QRS_idx-half_sample_size:middle_QRS_idx+half_sample_size])
        sample_list.append(label)
        if len(sample_list) == half_sample_size*2+1:
            sample_with_label.append(sample_list)
    np.savetxt(file_path, np.array(sample_with_label), fmt=my_fmt)
    
pass
