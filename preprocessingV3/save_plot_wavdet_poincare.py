# Check the result of wavedet

import numpy as np
import matplotlib.pyplot as plt
# import R_detection
# import R_refine1
import wfdb
# import time
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import scipy.io as scio
from statistics import median
# from math import floor
from mpl_toolkits.mplot3d import Axes3D
from iteration_utilities import deepflatten
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# Create custom colormaps
# 244, 246, 246 白
# 0.9569    0.9647    0.9647
# 0.0824    0.2627    0.3765
# 21, 101, 192 蓝
# 0.0824    0.3961    0.7529
# 255, 143, 0 橙
# 1.0000    0.5608         0
# 174, 213, 129绿
# 0.6824    0.8353    0.5059
# 191, 54, 12 红
# 0.7490    0.2118    0.0471
# 74, 20, 140 紫
# 0.2902    0.0784    0.5490


mycolor = np.array([[1.0000,    1.0000,    1.0000],
                    [0.0824,    0.3961,    0.7529],
                    [0.8000,    1.0000,    0.2000],
                    [1.0000,    0.5608,         0],
                    [0.7490,    0.2118,    0.0471],
                    [0.2902,    0.0784,    0.5490],])
red = mycolor[:,0]
green = mycolor[:,1]
blue = mycolor[:,2]

csegment = [0.0, 0.00001, 0.02, 0.1, 0.4, 1.0]

cdict = {'red':  ((csegment[0], 0, red[0]),   
                  (csegment[1], red[1], red[1]),   
                  (csegment[2], red[2], red[2]),
                  (csegment[3], red[3], red[3]),
                  (csegment[4], red[4], red[4]),
                  (csegment[5], red[5], 0)),  
        #
        'green': ((csegment[0], 0, green[0]),   
                  (csegment[1], green[1], green[1]),   
                  (csegment[2], green[2], green[2]),
                  (csegment[3], green[3], green[3]),
                  (csegment[4], green[4], green[4]),
                  (csegment[5], green[5], 0)),   
        #
        'blue':  ((csegment[0], 0, blue[0]),   
                  (csegment[1], blue[1], blue[1]),   
                  (csegment[2], blue[2], blue[2]),
                  (csegment[3], blue[3], blue[3]),
                  (csegment[4], blue[4], blue[4]),
                  (csegment[5], blue[5], 0)),  
        #
        'alpha': ((csegment[0], 0, 0),   
                  (csegment[1], 1, 1),   
                  (csegment[2], 1, 1),
                  (csegment[3], 1, 1),
                  (csegment[4], 1, 1),
                  (csegment[5], 1, 0)), 
        }  
#        row i:      x  y0  y1
#                          /
#                         /
#        row i+1:    x  y0  y1

cmap = LinearSegmentedColormap('Rd_Bl_Rd', cdict, 256)
# im = np.outer(np.ones(10), np.linspace(0, 255, 256))
# fig = plt.figure(figsize=(9, 2))
# ax = fig.add_subplot('111')
# ax.set_xticks(np.linspace(0, 255, 3))
# ax.set_xticklabels([0, 0.5, 1])
# ax.set_yticks([])
# ax.set_yticklabels([])
# ax.imshow(im, interpolation='nearest', cmap=cmap)
# plt.show()

def iu_deepflatten(a): 
    return list(deepflatten(a, depth=1)) 

def Read_ECG_and_ANN(position):
    """
    根据文件地址导入：ECG数据，注释，采样频率，记录名

    parameter:
        position(str): The position of record

    returns:
        list: ECG record
        list: Annotation ('0' stand for no apnea, '1' stand for apnea)
        int: The sampling frequency of ECG record
        str: The name of ECG record
    """
    temp = wfdb.rdrecord(position)
    ecg_data = temp.p_signal
    samp_freq = temp.fs
    reco_name = temp.record_name
    temp = wfdb.rdann(position, 'apn')
    anno_data = temp.symbol
    converter = {'N': 0, 'A': 1}
    return ([x[0] for x in ecg_data],
            [converter[x] for x in anno_data if x in ['N', 'A']],
            samp_freq,
            reco_name)


def R_Senior_Selection(QRS, ECG, boundary):
    """
    There will be a slightly error in R moment due to mean filter.
    Adjusting R_Moment_List by finding local maximum in a samll range
    to eliminate this error.

    Parameters:
        RSS_in(np.array): ECG data.
    """
    for idx, val in enumerate(QRS):
        max_index = np.argmax(np.array([ECG[x] for x in range(val-boundary, val+boundary)]))
        QRS[idx] = val-boundary+max_index
    return QRS




a = ['a'+str(x).zfill(2) for x in range(1,7)]
b = ['b'+str(x).zfill(2) for x in range(1,6)]
c = ['c'+str(x).zfill(2) for x in range(1,11)]
datalist = a + b + c
for DataID in datalist:
    ###### Acquire the ecg record and corresponding wavedet result from folder ######
    ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID
    QRSpath = ECGpath + '_QRS_detection'

    QRS = scio.loadmat(QRSpath)['wavedet_multilead']
    QRS = QRS[0][0][0].flatten()  # wavedet程序员是魔鬼吗
    ecg, annotation, sf, name = Read_ECG_and_ANN(ECGpath)


    QRS = R_Senior_Selection(QRS, ecg, 2)

    ###### 获得不同高度QRS左侧第一个和第二个RR间距 保存为 interval_features_list ######
    interval_num = 2
    interval_features_list = []
    for idx in range(len(QRS)):
        current_item = []
        try:
            #current_item: [QRS[idx1]-QRS[idx-1], QRS[idx1]-QRS[idx-2], ...]
            for n_left in range(interval_num):
                current_item.append(QRS[idx-n_left]-QRS[idx-n_left-1])
        except:
            # padding
            temp = QRS[idx+1] - QRS[idx]
            for n_left in range(interval_num):
                current_item.append(temp)
        interval_features_list.append(current_item)

    #根据sum去除异常值
    sum_of_double_RR_list = []
    for idx, val in enumerate(interval_features_list):
        sum_of_double_RR_list.append([sum(val), val])
    sum_of_double_RR_list.sort()
    margin = int(len(sum_of_double_RR_list)/1000)
    sum_of_double_RR_list = sum_of_double_RR_list[margin: -margin]
    new_interval_features_list = []
    for val in sum_of_double_RR_list:
        new_interval_features_list.append(val[1])


    #算出坐标
    new_interval_features_arr = np.array(new_interval_features_list)
    X = new_interval_features_arr[:,1]*10
    Y = new_interval_features_arr[:,0]*10

    #保存
    plt.cla()
    plt.scatter(X, Y, marker='.', alpha=0.3, s=5)
    plt.xlabel('RRn(ms)')
    plt.ylabel('RRn+1(ms)')
    plt.savefig("plot_wavedet_poincare/" + DataID + ".png")
    pass