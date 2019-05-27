import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import scipy.io as scio
from statistics import median
from mpl_toolkits.mplot3d import Axes3D
from iteration_utilities import deepflatten
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from scipy.stats import t, zscore


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

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


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
def get_ecg_sqrs125_wavedet(DataID, loacl_max_index=5):
    ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID
    QRSpath = ECGpath + '_QRS_detection'
    wavedet_ = scio.loadmat(QRSpath)['wavedet_multilead']
    wavedet_ = wavedet_[0][0][0].flatten()  # wavedet程序员是魔鬼吗
    ecg, sqrs125, annotation, sf, name = read_record(ECGpath)
    return (ecg,
            R_Senior_Selection(sqrs125, ecg, loacl_max_index),
            R_Senior_Selection(wavedet_, ecg, loacl_max_index),
            annotation)

def get_ecg_sqrs125_anno_sf(DataID, loacl_max_index=5):
    ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID
    ecg, sqrs125, annotation, sf, name = read_record(ECGpath)
    return (ecg,
            R_Senior_Selection(sqrs125, ecg, loacl_max_index),
            annotation,
            sf)

def get_ecg_wavedwet_anno_sf(DataID, loacl_max_index=5):
    ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID
    QRSpath = ECGpath + '_QRS_detection'
    wavedet_ = scio.loadmat(QRSpath)['wavedet_multilead']
    wavedet_ = wavedet_[0][0][0].flatten()  # wavedet程序员是魔鬼吗
    ecg, _, annotation, sf, name = read_record(ECGpath)
    return (ecg,
            R_Senior_Selection(wavedet_, ecg, loacl_max_index),
            annotation,
            sf)

###### 求QRS附近基线 - 裁判均值+中位数/2 ######
def get_baseline(QRS, ecg):
    baseline_list = []
    for val in QRS:
        segment = ecg[val-16: val-6] + ecg[val+6: val+16]
        base_mean = (sum(segment)-max(segment)-min(segment)) / len(segment)
        base_median = median(segment)
        baseline_list.append((base_median+base_mean)/2)
    return baseline_list

# def get_whole_baseline():
#     # baseline = []
#     # for val in ecg:
#     #     pass    
#     pass


def get_feature(QRS, ecg):
    # 计算基线
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
        interval_features_list.append(ecg[R_posi]-baseline)

    # 20分贝信噪比 wgn(item[:-1], SNR)
    temp = list(wgn(np.array(interval_features_list), 20))
    interval_features_list = []
    for item in temp:
        interval_features_list.append([item])

    ###### 获得QRS左右两侧RR间距 保存到 interval_features_list[:][1:2] ######
    QRS_padding = np.concatenate(([QRS[0]], QRS, [QRS[-1]]))
    for idx in range(1, len(QRS_padding)-1):
        current_item = [QRS_padding[idx]-QRS_padding[idx-1],
                        QRS_padding[idx+1]-QRS_padding[idx]]
        interval_features_list[idx-1].extend(current_item)
    return interval_features_list, baseline_list

def get_mean_height(interval_features_list):
    ###### 求出平均相对高度 ######
    interval_features_array =  np.array(interval_features_list)
    mean_height = np.mean(interval_features_array[:, 0])
    return mean_height

def get_trace(interval_features_list, mean_height):
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
                    try:
                        height_sum.append((interval_features_list[idx+1][0]-mean_height)**3)
                    except:
                        break
                    in_trace = True
            else:
                # 过去是非稳态
                if abs(trace_x-trace_y) < threshold:
                    stable_count+=1
                    if stable_count > 1:
                        # 累计两个稳态
                        stable = True  #认为过去是稳态
                        stable_count = 0
    return trace_list


def grubbs(X, test='max', alpha=0.05):
    """
    Performs Grubbs' test for outliers recursively until the null hypothesis is
    true.
    Parameters
    ----------
    X : ndarray
        A numpy array to be tested for outliers.
    test : str
        Describes the types of outliers to look for. Can be 'min' (look for
        small outliers), 'max' (look for large outliers), or 'two-tailed' (look
        for both).
    alpha : float
        The significance level.
    Returns
    -------
    X : ndarray
        The original array with outliers removed.
    outliers : ndarray
        An array of outliers.
    """

    Z = zscore(X, ddof=1)  # Z-score
    N = len(X)  # number of samples

    # calculate extreme index and the critical t value based on the test
    if test == 'two-tailed':
        extreme_ix = lambda Z: np.abs(Z).argmax()
        t_crit = lambda N: t.isf(alpha / (2.*N), N-2)
    elif test == 'max':
        extreme_ix = lambda Z: Z.argmax()
        t_crit = lambda N: t.isf(alpha / N, N-2)
    elif test == 'min':
        extreme_ix = lambda Z: Z.argmin()
        t_crit = lambda N: t.isf(alpha / N, N-2)
    else:
        raise ValueError("Test must be 'min', 'max', or 'two-tailed'")

    # compute the threshold
    thresh = lambda N: (N - 1.) / np.sqrt(N) * \
        np.sqrt(t_crit(N)**2 / (N - 2 + t_crit(N)**2))

    # create array to store outliers
    outliers = np.array([])

    # loop throught the array and remove any outliers
    del_index_list = []
    while abs(Z[extreme_ix(Z)]) > thresh(N):

        # update the outliers
        outliers = np.r_[outliers, X[extreme_ix(Z)]]
        # remove outlier from array
        X = np.delete(X, extreme_ix(Z))
        del_index_list.append(extreme_ix(Z))
        # repeat Z score
        Z = zscore(X, ddof=1)
        N = len(X)

    return X, outliers, del_index_list