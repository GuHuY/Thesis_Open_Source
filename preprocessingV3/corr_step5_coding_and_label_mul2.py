import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from pathlib import Path
from a_lib import *

half_sample_size = 30
my_fmt = '%d'
for idx in range(0, half_sample_size*2):
    my_fmt += ' %d'

a = ['a'+str(x).zfill(2) for x in range(1,21)]
b = ['b'+str(x).zfill(2) for x in range(1,6)]
c = ['c'+str(x).zfill(2) for x in range(1,11)]
datalist = a + b + c
for DataID in datalist:
    ecg, wavedet_, annotation, samp_freq = get_ecg_wavedwet_anno_sf(DataID, 5)
    QRS = wavedet_
    # ecg, sqrs125, annotation, samp_freq = get_ecg_wavedwet_anno_sf(DataID, 5)
    # QRS = sqrs125
    features_list, baseline_list = get_feature(QRS, ecg)
    features_arr = np.array(features_list)
    QRS_arr = np.array(QRS)

    # 找到稳态吸引子核心
    R = features_arr[:,1:3]*10
    bandwidth = estimate_bandwidth(R, quantile=0.47, n_samples=1300)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(R)  # 训练模型
    labels = ms.labels_  # 所有点的的labels
    cluster_centers = ms.cluster_centers_  # 聚类得到的中心点
    cluster_center = cluster_centers[0]
    RR_mean = (cluster_center[0]+cluster_center[1]) / 2
    RR_Border_Low = 0.4 * RR_mean/10
    RR_Border_High = 2 * RR_mean/10

    # 拒绝域
    del_idx_list = []
    for idx, val in enumerate(features_list):
        del_sign = True
        RR1 = val[1]
        RR2 = val[2]
        if RR1 > RR_Border_Low and RR1 < RR_Border_High and RR2 > RR_Border_Low and RR2 < RR_Border_High:
            del_sign = False
        if del_sign:
            del_idx_list.extend([idx-1, idx, idx+1])
    del_idx_list = list(set(del_idx_list))
    features_arr = np.delete(features_arr, del_idx_list, axis=0)
    QRS_arr = np.delete(QRS_arr, del_idx_list)

    #修正H为H'
    Z = []
    temp = list(features_arr[:, 0])
    temp = [temp[0]] + temp + [temp[-1]]
    for idx in range(1, len(temp)-1):
        Z.append(min([temp[idx-1], temp[idx], temp[idx+1]]))
    Z = np.array(Z)

    # 根据H'排除P波
    del_idx_list = []
    features_list = list(features_arr)
    H_border = get_mean_height(features_list) * 0.25
    for idx, val in enumerate(features_list):
        if Z[idx] < H_border:
            # del_idx_list.extend([idx-1, idx, idx+1])
            del_idx_list.append(idx)
    del_idx_list = list(set(del_idx_list))
    features_arr = np.delete(features_arr, del_idx_list, axis=0)
    QRS_arr = np.delete(QRS_arr, del_idx_list)


    # 指标
    Indicator = features_arr[:,1] + features_arr[:,2] 
    
    # 编码----

    #----

    file_path = '/Users/rex/python/z_thesis/RR_trace_wavedet/' + DataID + '.txt'
    # file_path = '/Users/rex/python/z_thesis/RR_trace_sqrs/' + DataID + '.txt'
    path = Path(file_path)
    if path.exists():
        print('update ', end="")
        path.unlink()  # delete file
    print(DataID)
    # Short_Label_Sign = False

    T = samp_freq * 60
    QRS_list = list(QRS_arr)
    QL_len = len(QRS_list)
    wedge = 0
    sample_with_label = []
    for idx, label in enumerate(annotation):
        middle_Time = int(idx * T + T/2)
        for i in range(wedge, QL_len):
            if QRS_list[i] > middle_Time:
                middle_QRS_idx = wedge = i
                break
        sample_list = list(Indicator[middle_QRS_idx-half_sample_size:middle_QRS_idx+half_sample_size])
        sample_list.append(label)
        if len(sample_list) == half_sample_size*2+1:
            sample_with_label.append(sample_list)
    np.savetxt(file_path, np.array(sample_with_label), fmt=my_fmt)
pass
