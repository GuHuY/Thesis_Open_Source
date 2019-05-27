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

normal = 800
interval_list = [normal, normal, 283, normal-283, normal, normal]
points = []
for idx in range(len(interval_list)-1):
    points.append([interval_list[idx], interval_list[idx+1]])
points = np.array(points)

normal = 800
interval_list_2 = [normal, normal, normal-150, 150+normal, normal, normal]
points_2 = []
for idx in range(len(interval_list_2)-1):
    points_2.append([interval_list_2[idx], interval_list_2[idx+1]])
points_2 = np.array(points_2)

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

a = ['a'+str(x).zfill(2) for x in range(1,21)]
b = ['b'+str(x).zfill(2) for x in range(1,6)]
c = ['c'+str(x).zfill(2) for x in range(1,11)]
datalist = a + b + c
###### Acquire the ecg record and corresponding wavedet result from folder ######
DataID = 'a09'
ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID

QRSpath = ECGpath + '_QRS_detection'
wavedet_ = scio.loadmat(QRSpath)['wavedet_multilead']
wavedet_ = wavedet_[0][0][0].flatten()  # wavedet程序员是魔鬼吗

ecg, sqrs125, annotation, sf, name = read_record(ECGpath)

QRS = R_Senior_Selection(wavedet_, ecg, 2)

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
plt.figure()
plt.scatter(X, Y, marker='.', alpha=0.3, s=5)
# plt.plot(points[:,0], points[:,1], c='g')
# plt.plot(points[:,0], points[:,1], c='c', marker='o')
# for i in range(len(points)-1):
#     plt.text(points[i,0], points[i,1]+30, str(i+1))
# plt.text(points[-1,0], points[-1,1]-70, str(len(points)))


plt.plot(points_2[:,0], points_2[:,1], c='g')
plt.plot(points_2[:,0], points_2[:,1], c='g', marker='o')
for i in range(len(points_2)-1):
    plt.text(points_2[i,0], points_2[i,1]+30, str(i+1))
plt.text(points_2[-1,0], points_2[-1,1]-70, str(len(points_2)))



plt.xlabel('RRn(ms)')
plt.ylabel('RRn+1(ms)')
plt.show()