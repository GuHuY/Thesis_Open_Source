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

csegment = [0.0, 0.125, 0.25, 0.5, 0.7, 1.0]

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
        'alpha': ((csegment[0], 0, 0.5),   
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

def read_record(position):
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
    # ECG record
    temp = wfdb.rdrecord(position)
    ecg_data = [x[0] for x in temp.p_signal]
    samp_freq = temp.fs
    reco_name = temp.record_name

    # Qrs record
    temp = wfdb.rdann(position, 'qrs')
    qrs_data = temp.sample + np.array([4]*len(temp.sample))

    # Apnea annotations record
    temp = wfdb.rdann(position, 'apn')
    converter = {'N': 0, 'A': 1}
    anno_data = [converter[x] for x in temp.symbol if x in ['N', 'A']]
    return (ecg_data, qrs_data, anno_data, samp_freq, reco_name)


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


###### Acquire the ecg record and corresponding wavedet result from folder ######
DataID = 'a17'
ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID
QRSpath = ECGpath + '_QRS_detection'

wavedet_ = scio.loadmat(QRSpath)['wavedet_multilead']
wavedet_ = wavedet_[0][0][0].flatten()  # wavedet程序员是魔鬼吗
ecg, sqrs125, annotation, sf, name = read_record(ECGpath)


QRS = R_Senior_Selection(sqrs125, ecg, 5)

###### 求QRS附近基线 - 裁判均值+中位数/2 ######
baseline_list = []
for val in QRS:
    segment = ecg[val-16: val-6] + ecg[val+6: val+16]
    base_mean = (sum(segment)-max(segment)-min(segment)) / len(segment)
    base_median = median(segment)
    baseline_list.append((base_median+base_mean)/2)

#特征储存list 格式为[[Height, Interval_left_1, Interval_left_2],
#                  ...]
interval_features_list = []

###### 获得基线到QRS高度差 保存到 interval_features_list[:][0] ######
for R_posi, baseline in zip(QRS, baseline_list):
    interval_features_list.append([ecg[R_posi]-baseline])

###### 获得QRS左右两侧RR间距 保存到 interval_features_list[:][1:2] ######
interval_features_list_idx = 0
for idx in range(len(QRS)):
    current_item = []
    try:
        current_item = [QRS[idx]-QRS[idx-1], QRS[idx+1]-QRS[idx]]
    except:
        # padding
        try:
            current_item = [QRS[idx]-QRS[idx-1], QRS[idx]-QRS[idx-1]]
        except:
            current_item = [QRS[idx+1]-QRS[idx], QRS[idx+1]-QRS[idx]]
    interval_features_list[interval_features_list_idx].extend(current_item)
    interval_features_list_idx += 1


###### 求出平均相对高度 ######
interval_features_array =  np.array(interval_features_list)
mean_height = np.mean(interval_features_array[:, 0])
    
###### 遍历 interval_features_list 获取全部轨迹 保存为trace_dict ######
threshold = 4 # 单位为10ms, 即误差80ms,小于PR间期
trace_list = []  # [idx, posi, trace_length, average_height]
trace_head = 0
in_trace = False
height_sum = []
posi = []
trace_length = 0
stable = False
stable_count = 0
for idx, val in enumerate(interval_features_list):
    trace_x = val[1]
    trace_y = val[2]
    if in_trace:
        #在轨迹中
        if trace_length<10:
            # 轨迹长度正常
            if abs(trace_x-trace_y) >= threshold :
                #继续记录
                posi.append([trace_x, trace_y])
                if trace_length < 2:
                    height_sum.append((interval_features_list[idx+1][0]-mean_height)**3)
                trace_length += 1
            else:
                #轨迹结束
                trace_list.append([trace_head, posi, trace_length, sum(height_sum)/len(height_sum)])
                in_trace = False
                stable = True
                posi = []
                trace_length = 0
                height_sum = []
        else:
            # 轨迹长度异常
            # 清空轨迹
            in_trace = False
            stable = False
            posi = []
            trace_length = 0
            height_sum = []
    else:
        #不在轨迹中
        if stable:
            # 过去是稳态
            if abs(trace_x-trace_y) >= threshold: 
                #轨迹开始
                trace_head = idx #+ 1
                posi.append([trace_x, trace_y])
                trace_length += 1
                height_sum.append((interval_features_list[idx+1][0]-mean_height)**3)
                in_trace = True
        else:
            # 过去是非稳态
            if abs(trace_x-trace_y) < threshold:
                stable_count+=1
                if stable_count > 1:
                    # 累计两个稳态
                    stable = True  #认为过去是稳态
                    stable_count = 0



#算出坐标
X = []
Y = []
Z = []
# item: [idx, posi, trace_length, average_height]
for item in trace_list:
    for posi in item[1]:
        X.append(posi[0])
        Y.append(posi[1])
        Z.append(item[3])
X = np.array(X)*10
Y = np.array(Y)*10
Z = np.array(Z)

# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
# cmap=plt.get_cmap('rainbow')
m_sactter = ax.scatter(X, Y, Z, c=Z, cmap=plt.get_cmap('rainbow'), s=1)
fig.colorbar(m_sactter, shrink=0.5)
ax.set_xlim(0, 1.1*X.max())
ax.set_ylim(0, 1.1*Y.max())
ax.set_zlim(1.1*Z.min(), 1.1*Z.max())
# ax.plot(Y, Z, 'k.', zdir='x', zs=1.1*X.max())
# ax.plot(X, Z, 'k.', zdir='y', zs=1.1*Y.max())
# ax.plot(X, Y, 'k.', zdir='z', zs=1.1*Z.min())


# ax.legend(loc='best')
 
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.view_init(90, 270)
# ax.view_init(30, 270-45)
# ax.view_init(30, 45)

plt.show()

pass