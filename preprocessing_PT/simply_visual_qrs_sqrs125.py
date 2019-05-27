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

# def my_round(a):
#     integer = int(a//1)
#     decimal = a-integer
#     if decimal < 0.5: return integer
#     else: return integer+1

###### Acquire the ecg record and corresponding wavedet result from folder ######
DataID = 'a04'
ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID
# QRSpath = ECGpath + '_QRS_detection'

# QRS = scio.loadmat(QRSpath)['wavedet_multilead']
# QRS = QRS[0][0][0].flatten()  # wavedet程序员是魔鬼吗
ecg, sqrs125, annotation, sf, name = read_record(ECGpath)

# QRS = R_Senior_Selection(QRS, ecg, 2)
QRS = R_Senior_Selection(sqrs125, ecg, 4)

###### 求QRS附近基线 - 均值+中位数/2 ######
# baseline_list = []
# for val in QRS:
#     segment = ecg[val-16: val-6] + ecg[val+6: val+16]
#     base_mean = (sum(segment)-max(segment)-min(segment)) / len(segment)
#     base_median = median(segment)
#     baseline_list.append((base_median+base_mean)/2)


plt.figure()

# ecg_x = np.linspace(0,len(ecg)*10,len(ecg),endpoint=False)
# ecg = np.array(ecg)
# sqrs125_x = sqrs125 * 10
# 
# plt.plot(QRS,
#          [ecg[t] for t in QRS],
#          'x',
#          label='wavedet')
# plt.plot(QRS, baseline_list, 'o', label='baseline')
# plt.plot(sqrs125_x,
#          [ecg[t] for t in sqrs125],
#          'x',
#          label='QRS from .qrs')

plt.plot(ecg, label=DataID)
plt.plot(QRS,
         [ecg[t] for t in QRS],
         'x',
         label='QRS from .qrs')


plt.xlabel('Time(0.01s)')
plt.ylabel('Voltage(v)')
plt.legend(loc='upper right')
plt.show()
pass