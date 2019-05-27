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


###### Acquire the ecg record and corresponding wavedet result from folder ######
DataID = 'a19'
ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID
QRSpath = ECGpath + '_QRS_detection'

QRS = scio.loadmat(QRSpath)['wavedet_multilead']
QRS = QRS[0][0][0].flatten()  # wavedet程序员是魔鬼吗
ecg, annotation, sf, name = Read_ECG_and_ANN(ECGpath)


# QRS = R_Senior_Selection(QRS, ecg, 2)


###### 求QRS附近基线 - 均值+中位数/2 ######
baseline_list = []
for val in QRS:
    segment = ecg[val-16: val-6] + ecg[val+6: val+16]
    base_mean = (sum(segment)-max(segment)-min(segment)) / len(segment)
    base_median = median(segment)
    baseline_list.append((base_median+base_mean)/2)

###### 获得基线到QRS高度差 保存为height_diff_list ######
height_diff_list = []
for R_posi, baseline in zip(QRS, baseline_list):
    height_diff_list.append(ecg[R_posi]-baseline)

###### 统计height_diff_list的分布 保存为hist 对应的电压在bin_centers ######
bin_num, bin_width, centre = 421, 0.1, 0
hist_bias = bin_num / 2 * bin_width
lins_bias = (bin_num-1) / 2 * bin_width
hist, bin_left_edges = np.histogram(np.array(height_diff_list),
                                    bins=bin_num,
                                    range=(centre-hist_bias, centre+hist_bias),
                                    density=True)
hist = hist/sum(hist) * 100
bin_centers = bin_left_edges[:-1] + bin_width / 2
# statistics = interp1d(bin_centers, hist, kind='quadratic')


###### 获得不同高度QRS左右两侧RR间距和 保存为biRR_height_dict ######
# biRR_height_dict = {}
# # initialize biRR_height
# for val in bin_centers:
#     biRR_height_dict[round(val, 1)] = [] # biRR_height_dict.keys()=[-11.0, -10.9, ..., 11.0]

# for idx in range(len(QRS)):
#     # Get biRR
#     try:
#         biRR = QRS[idx+1] - QRS[idx-1]
#     except:
#         # padding
#         if idx: biRR = (QRS[idx]-QRS[idx-1]) * 2  # last one
#         else: biRR = (QRS[idx+1]-QRS[idx]) * 2  # first one
#     # Get bin
#     height = height_diff_list[idx]
#     # 下面一行待验证
#     corr_bin = ((height-centre+bin_width/2)//bin_width) * bin_width
#     biRR_height_dict[round(corr_bin, 1)].append(biRR)

###### 获得不同高度QRSy右单层侧RR间距 保存为biRR_height_dict ######
biRR_height_dict = {}
# Initialize biRR_height_dict
for val in bin_centers:
    biRR_height_dict[round(val, 1)] = [] # biRR_height_dict.keys()=[-11.0, -10.9, ..., 11.0]

for idx in range(len(QRS)):
    try:
        biRR = QRS[idx+1] - QRS[idx]
    except:
        # padding
        biRR = QRS[idx] - QRS[idx-1]  # last one
    # 获取对应高度,进而计算出对应的bin作为key,将当前RRinterval作为value
    height = height_diff_list[idx]
    corr_bin = ((height-centre+bin_width/2)//bin_width) * bin_width
    biRR_height_dict[round(corr_bin, 1)].append(biRR)

###### 统计biRR_height_dict中每个value的分布 ######
height_bin_biRR_hist = []
#去除过大和过小的值来保证min_biRR和max_biRR合适,margin为双边1/1000
values_from_biRR_height_dict = iu_deepflatten([x for _, x in enumerate(biRR_height_dict.values())])
values_from_biRR_height_dict.sort()
margin = int(len(values_from_biRR_height_dict)/1000)
min_biRR = values_from_biRR_height_dict[margin]
max_biRR = values_from_biRR_height_dict[-margin]
#进行统计
for item in biRR_height_dict.values():
    curr_hist, curr_bin_left = np.histogram(np.array(item),
                                        bins=101,
                                        range=(min_biRR, max_biRR),
                                        density=False)
    height_bin_biRR_hist.append(curr_hist)


fig = plt.figure()
ax = Axes3D(fig)
# X:基线到QRS高度差
# 控制X的异常值,防止3D图像X轴方向过密
exists_hist_idx_list = [] 
for idx, val in enumerate(hist):
    if val > 0.001:
        exists_hist_idx_list.append(idx)
X = bin_centers[min(exists_hist_idx_list): max(exists_hist_idx_list)]
# Y:QRS左右两侧RR间距和的分布
# 统计Y时已经做了margin
Y = curr_bin_left[:-1] + (max_biRR-min_biRR) / (101*2)
# Z
X_bin_head = centre - ((bin_num-1)/2) * bin_width
Y_bin_width = (max_biRR-min_biRR) / (101-1)

Z = []
for y in (Y-min_biRR)//Y_bin_width:
    temp = []
    for x in (X-X_bin_head)//bin_width:
        temp.append(height_bin_biRR_hist[int(x)][int(y)])
    Z.append(temp)
Z  = np.array(Z)

X_p, Y_p = np.meshgrid(X, Y)    # x-y 平面的网格
# CMRmap, gist_rainbow, rainbow, !ocean, !!nipy_spectral, !!gnuplot, !!!gist_stern, 
# cmap = plt.get_cmap('rainbow')
ax.contourf(X_p, Y_p, Z, zdir='y', offset=min(Y)-(max(Y)-min(Y))/10, cmap=cmap, shade=True)
ax.contourf(X_p, Y_p, Z, zdir='x', offset=min(X)-(max(X)-min(X))/10, cmap=cmap, shade=True)
# ax.contourf(X_p, Y_p, Z, zdir='z', offset=-Z.max()/3, cmap=cmap) 
surf = ax.plot_surface(X_p, Y_p, Z, rstride=1, cstride=1, cmap=cmap, shade=False)
ax.set_zlim(0, 1.1*Z.max()) 
ax.set_xlabel('Height')
ax.set_ylabel('RRintervals Sum')
ax.set_zlabel('      Quantity')
fig.colorbar(surf, shrink=0.5)
# ax.view_init(90, 90)
ax.view_init(60, 45)
# plt.title(DataID)
plt.show()

#---------- display statistics ----------#
# statistics_x = np.linspace(-5, 10, 1000, endpoint=False)
# statistics_y = statistics(statistics_x)
# plt.plot(statistics_x, statistics_y)
# plt.savefig("plot_storage/" + DataID + ".png")
#----------------------------------------#


pass