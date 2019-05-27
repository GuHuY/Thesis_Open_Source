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
for DataID in datalist:
    ###### Acquire the ecg record and corresponding wavedet result from folder ######
    ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID
    ecg, sqrs125, annotation, sf, name = read_record(ECGpath)


    QRS = R_Senior_Selection(sqrs125, ecg, 5)



    ###### 求QRS附近基线 - 均值+中位数/2 ######
    baseline_list = []
    for val in QRS:
        segment = ecg[val-16: val-6] + ecg[val+6: val+16]
        base_mean = np.mean(np.array(segment))
        base_median = median(segment)
        baseline_list.append((base_median+base_mean)/2)

    ###### 统计基线到QRS高度差 ######
    height_diff_list = []
    for R_posi, baseline in zip(QRS, baseline_list):
        height_diff_list.append(ecg[R_posi]-baseline)

    bin_num, bin_width, centre = 221 , 0.1 , 0
    hist_bias = bin_num / 2 * bin_width
    lins_bias = (bin_num-1) / 2 * bin_width
    hist, bin_left_edges = np.histogram(np.array(height_diff_list),
                                        bins=bin_num,
                                        range=(centre-hist_bias, centre+hist_bias),
                                        density=True)
    hist = hist/sum(hist) * 100
    bin_centers = bin_left_edges[:-1] + bin_width / 2
    statistics = interp1d(bin_centers, hist, kind='quadratic')

    #---------- display statistics ----------#
    statistics_x = np.linspace(-10, 10, 2000, endpoint=False)
    statistics_y = statistics(statistics_x)
    plt.cla()
    plt.plot(statistics_x, statistics_y)
    plt.savefig("sqrs125_height/" + DataID + ".png")
    #----------------------------------------#

