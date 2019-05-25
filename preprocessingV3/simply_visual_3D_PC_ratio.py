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
mycolor = np.array([[1.0000,    1.0000,    1.0000],  # 白
                    [0.0824,    0.3961,    0.7529],  # 蓝
                    [0.8000,    1.0000,    0.2000],  # 荧黄
                    [1.0000,    0.5608,         0],  # 橙
                    [0.7490,    0.2118,    0.0471],  # 红
                    [0.2902,    0.0784,    0.5490],  # 紫
                    ])
red = mycolor[:,0]
green = mycolor[:,1]
blue = mycolor[:,2]
csegment = [0.0, 0.00001, 0.02, 0.1, 0.4, 1.0]  # 颜色分布 
        # row i:      x            y0     y1
        #                                /
        #                               /
        #                              /
        #                             /
        # row i+1:    x            y0     y1
cdict = {'red':  ((csegment[0], 0,      red[0]),   
                  (csegment[1], red[1], red[1]),   
                  (csegment[2], red[2], red[2]),
                  (csegment[3], red[3], red[3]),
                  (csegment[4], red[4], red[4]),
                  (csegment[5], red[5], 0)),  
        #
        'green': ((csegment[0], 0,        green[0]),   
                  (csegment[1], green[1], green[1]),   
                  (csegment[2], green[2], green[2]),
                  (csegment[3], green[3], green[3]),
                  (csegment[4], green[4], green[4]),
                  (csegment[5], green[5], 0)),   
        #
        'blue':  ((csegment[0], 0,       blue[0]),   
                  (csegment[1], blue[1], blue[1]),   
                  (csegment[2], blue[2], blue[2]),
                  (csegment[3], blue[3], blue[3]),
                  (csegment[4], blue[4], blue[4]),
                  (csegment[5], blue[5], 0)),  
        #
        'alpha': ((csegment[0], 0, 0),   # 透明
                  (csegment[1], 1, 1),   
                  (csegment[2], 1, 1),
                  (csegment[3], 1, 1),
                  (csegment[4], 1, 1),
                  (csegment[5], 1, 0)), 
        }  

cmap = LinearSegmentedColormap('w_b_y_o_r_p', cdict, 256)


def iu_deepflatten(a): 
    """
    Flatten larger scale 2D array.

    Parameter:
        a(np.array): 2D array.
    
    Return:
        (mp.array): 1D array.
    """
    return list(deepflatten(a, depth=1)) 

def Read_ECG_and_ANN(position):
    """
    Import ECG data, annotation, sampling frequency and record name.

    Parameter:
        position(str): The position of record

    Returns:
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
    Adjusting R-peak position to local maximum in a samll range.

    Parameters:
        QRS(np.array): R peak position.
        ECG(list): ECG data.
        boundary(int): range = (QRS[n]-boundary, QRS[n]+boundary)
    
    Return:
        (np.array): adjusted QRS array.
    """
    for idx, val in enumerate(QRS):
        max_index = np.argmax(np.array([ECG[x] for x in range(val-boundary, val+boundary)]))
        QRS[idx] = val-boundary+max_index
    return QRS


###### Acquire the ecg record and corresponding wavedet result from folder ######
DataID = 'a02'
ECGpath = '/Users/rex/Documents/MATLAB/ecg_kit/ecg-kit/raw_ECG/' + DataID
QRSpath = ECGpath + '_QRS_detection'

QRS = scio.loadmat(QRSpath)['wavedet_multilead']
QRS = QRS[0][0][0].flatten()  # wavedet程序员是魔鬼吗
ecg, annotation, sf, name = Read_ECG_and_ANN(ECGpath)


# QRS = R_Senior_Selection(QRS, ecg, 2)


###### 求 QRS 附近基线 - 裁判均值+中位数/2 ######
baseline_list = []
for val in QRS:
    segment = ecg[val-16: val-6] + ecg[val+6: val+16]
    base_mean = (sum(segment)-max(segment)-min(segment)) / len(segment)
    base_median = median(segment)
    baseline_list.append((base_median+base_mean)/2)

###### 获得基线到QRS高度差 保存为 height_diff_list ######
height_diff_list = []
for R_posi, baseline in zip(QRS, baseline_list):
    height_diff_list.append(ecg[R_posi]-baseline)

###### 统计 height_diff_list 的分布 保存为 hist 对应的电压在 bin_centers ######
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


###### 获得QRS到其左侧一个,两个,n ..., n个QRS的距离之间的比值 保存为 interval_features_list ######
interval_num = 3
interval_features_list = []
for idx in range(len(QRS)):
    current_item = []
    try:
        #current_item: [QRS[idx1]-QRS[idx-1], QRS[idx1]-QRS[idx-2], ...]
        for n_left in range(1, interval_num+1):
            current_item.append(QRS[idx]-QRS[idx-n_left])
    except:
        # padding
        temp = QRS[idx+1] - QRS[idx]
        for n_left in range(1, interval_num+1):
            current_item.append(n_left*temp)
    ratio_list = []
    for i_current_item, v_current_item in enumerate(current_item):
        if i_current_item > 0:
            ratio_list.append(v_current_item / current_item[i_current_item-1])
    interval_features_list.append(ratio_list)

###### 进行PCA将特征降到1维 ######
pca = PCA(n_components=1)
new_feature = pca.fit_transform(np.array(interval_features_list))
print('explained_variance_ratio_:', pca.explained_variance_ratio_)
print('components_:', pca.components_)

###### 获得不同高度QRS的new_feature 保存为 Key_height_Val_feature_dict ######
Key_height_Val_feature_dict = {}
# Initialize Key_height_Val_feature_dict
for val in bin_centers:
    Key_height_Val_feature_dict[round(val, 1)] = [] # biRR_height_dict.keys()=[-11.0, -10.9, ..., 11.0]
for idx in range(len(QRS)):
    # Get bin
    height = height_diff_list[idx]
    corr_bin = ((height-centre+bin_width/2)//bin_width) * bin_width
    Key_height_Val_feature_dict[round(corr_bin, 1)].append(new_feature[idx][0])

###### 统计biRR_height_dict中每个value的分布 ######
height_bin_biRR_hist = []
#去除过大和过小的值来保证min_biRR和max_biRR合适,margin为双边1/1000
values_from_biRR_height_dict = iu_deepflatten([x for _, x in enumerate(Key_height_Val_feature_dict.values())])
values_from_biRR_height_dict.sort()
margin = int(len(values_from_biRR_height_dict)/1000)
min_biRR = values_from_biRR_height_dict[margin]
max_biRR = values_from_biRR_height_dict[-margin]
#进行统计
for item in Key_height_Val_feature_dict.values():
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
# cmap = plt.get_cmap('coolwarm')
ax.contourf(X_p, Y_p, Z, zdir='y', offset=min(Y)-(max(Y)-min(Y))/10, cmap=cmap, shade=True)
ax.contourf(X_p, Y_p, Z, zdir='x', offset=min(X)-(max(X)-min(X))/10, cmap=cmap, shade=True)
# ax.contourf(X_p, Y_p, Z, zdir='z', offset=-Z.max()/3, cmap=cmap) 
surf = ax.plot_surface(X_p, Y_p, Z, rstride=1, cstride=1, cmap=cmap, shade=False)
ax.set_zlim(0, 1.1*Z.max()) 
ax.set_xlabel('Height')
ax.set_ylabel('RRinterval-PCA')
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