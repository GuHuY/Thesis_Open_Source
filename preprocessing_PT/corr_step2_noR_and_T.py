import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from a_lib import *


ecg, sqrs125, wavedet_ = get_ecg_sqrs125_wavedet('c03', 5)
features_list, baseline_list = get_feature(sqrs125, ecg)
H_mean = get_mean_height(features_list)
trace_list = get_trace(features_list, H_mean)
features_arr = np.array(features_list)

X = features_arr[:,1:3]*10

# #############################################################################
# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.47, n_samples=1300)
# print('bandwidth = ', bandwidth)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)  # 训练模型
labels = ms.labels_  # 所有点的的labels
cluster_centers = ms.cluster_centers_  # 聚类得到的中心点
cluster_center = cluster_centers[0]
RR_mean = (cluster_center[0]+cluster_center[1]) / 2
RR_Border_Low = 0.4 * RR_mean/10
RR_Border_High = 2 * RR_mean/10

print(cluster_center, RR_mean, RR_Border_Low ,RR_Border_High)

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


plt.figure()
plt.scatter(features_arr[:,1]*10, features_arr[:,2]*10, marker='.', alpha=0.3, s=5)
plt.xlabel('RRn(ms)')
plt.ylabel('RRn+1(ms)')
plt.show()
