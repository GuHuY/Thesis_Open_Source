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


def sigmoid(x):
    s = 1 / (1 + np.exp(-x/k))
    return s

def Dimensionality_Reduction_PCA(data):
    pca=PCA(n_components='mle')
    newData=pca.fit_transform(data)
    return newData

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


# def visualize_histogram(d, DataID):
#     plt.figure()
#     n, bins, patches = plt.hist(x=d, bins=40, color='#0504aa', range=(-1,1), density=True,
#                                 alpha=0.7, rwidth=0.85)
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('Votage')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of ' + DataID)
#     plt.text(23, 45, r'$\mu=15, b=3$')
#     # maxfreq = n.max()
#     # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#     print(sum(n))
#     print(bins)
#     plt.show()


def generate_max_min_dist(posi, boundary):
    """
    Parameters:
        posi(int): position in ecg
        boundary(int): range = [posi-boundary: posi+boundary]
        ECG(list): ecg list
    Returns:
        int: max value in range
        int: min value in range
        int: distance between max and min
    """
    array = np.array(ecg[posi-boundary: posi+boundary])
    max_index = np.argmax(array)
    min_index = np.argmin(array)
    return (ecg[posi-boundary+max_index],
            ecg[posi-boundary+min_index], 
            abs(max_index-min_index))

def Evaluation(Before, frequency=6):
    """
    parameters:
        Before(dict): The output of B()
        frequency(int): The sampling frequency apply on B

    return:
        list: [[X of sampling result], [Y of sampling result]]
    """
    x_before_samp, y_before_samp = Before.keys(), Before.values()
    fl = interp1d(x_before_samp, y_before_samp, kind='quadratic')
    # 采样间隔设置
    x_after_samp = np.linspace(min(x_before_samp),
                               max(x_before_samp//1000*1000),
                               int(max(x_before_samp)//1000*frequency),
                               endpoint=False)
    # 采样
    y_after_samp = fl(x_after_samp)
    return [[int(round(x)) for x in x_after_samp], y_after_samp]

###### Acquire the ecg record and corresponding wavedet result from folder ######
DataID = 'a08'
ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID
QRSpath = ECGpath + '_QRS_detection'

QRS = scio.loadmat(QRSpath)['wavedet_multilead']
QRS = QRS[0][0][0].flatten()  # wavedet程序员是魔鬼吗
ecg, annotation, sf, name = Read_ECG_and_ANN(ECGpath)


QRS = R_Senior_Selection(QRS, ecg, 2)

###### Generate histogram of ecg (start from -5 to 5 with 0.1 bin size) ######
bin_num, bin_width, centre = 221 , 0.1 , 0
# bin_num must be odd
# eg. 3, 1, 2 denote: [0.5, 1.5], [1.5, 2.5], [2.5, 3.5]
hist_bias = bin_num / 2 * bin_width
lins_bias = (bin_num-1) / 2 * bin_width
hist, bin_left_edges = np.histogram(np.array(ecg),
                                    bins=bin_num,
                                    range=(centre-hist_bias, centre+hist_bias),
                                    density=True)
hist = hist/sum(hist) * 100
bin_value = list(hist)
for idx in range(1, int((bin_num+1)/2)):
    bin_value[idx] += bin_value[idx-1]
for idx in range(bin_num-2, int((bin_num-1)/2), -1):
    bin_value[idx] += bin_value[idx+1]
bin_centers = bin_left_edges[:-1] + bin_width / 2
statistics = interp1d(bin_centers, bin_value, kind='quadratic')
#---------- display hist ----------#
# new_dict = {}
# hist = np.around(hist, decimals=3)
# np.set_printoptions(suppress=True)
# bin_left_edges = np.around(bin_left_edges, decimals=2)
# for idx, val in enumerate(hist):
#     new_dict[bin_centers[idx]] = val
# for keys in new_dict.keys():
#     print(keys, '\t', new_dict[keys])
#----------------------------------#

#---------- display statistics ----------#
# statistics_x = np.linspace(-5, 5, 100, endpoint=False)
# statistics_y = statistics(statistics_x)
# plt.figure()
# plt.plot(statistics_x, statistics_y)
# plt.show()
#----------------------------------------#

###### Grading each QRS ###### 
features = []
features_pca = []
features_lib = {}
for R_posi in QRS:
    max_val, min_val, dist = generate_max_min_dist(R_posi, 6)
    features_lib[R_posi]=([float(statistics(ecg[R_posi])),
                           float(statistics(min_val)),
                           float(statistics(max_val)),
                           dist])
    features_pca.append([float(statistics(ecg[R_posi])),
                         float(statistics(min_val)),
                         float(statistics(max_val)),
                         float(statistics(min_val))*float(statistics(max_val))*dist,
                         dist])
    features.append(float(statistics(min_val))*float(statistics(max_val))*dist)



###### Sigmoid adjust
features_mean = np.mean(features)
k = features_mean/5
for idx, val in enumerate(features):
    features_pca[idx][3] = features[idx] = sigmoid(val)

###### 如果feature大于一定值排除该QRS
delete_QRS = []
for idx, val in enumerate(features):
    if val>0.85:
        delete_QRS.append(QRS[idx])



###### Rule like adjust
# features_mean = np.mean(features)
# k = features_mean/5
# for idx, val in enumerate(features):
#     if val > features_mean:
#         features[idx] = 1
#     else: 
#         features[idx] = val / features_mean


# print(features[:20])
# features = np.array(features)
print(features[:20])
pca=PCA(n_components=2)
newData=pca.fit_transform(np.array(features_pca))



plt.figure()
# plt.plot(newData[:, 0],newData[:, 1], '.')
plt.scatter(newData[:, 0],newData[:, 1], alpha=0.1, marker='o')
plt.show()



plt.figure()
# plt.plot(newData[:, 0],newData[:, 1], '.')
plt.scatter(newData[:, 0],newData[:, 1], alpha=0.01, marker='o')
plt.show()



plt.figure()
plt.plot(ecg)
plt.plot(QRS,
         [ecg[t] for t in QRS],
         'x',
         label='R wave')
plt.plot(QRS, features, 'o')
plt.plot(delete_QRS, [ecg[t] for t in delete_QRS], 'x')
# for idx, val in enumerate(QRS):
#     plt.text(val, ecg[val], features[idx])
plt.show() 

pass