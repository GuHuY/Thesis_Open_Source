import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from a_lib import *


ecg, sqrs125, wavedet_ = get_ecg_sqrs125_wavedet('a01', 5)
features_list, baseline_list = get_feature(wavedet_, ecg)
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
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    print(cluster_center)
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.xlabel('RRn(ms)')
plt.ylabel('RRn+1(ms)')
plt.title('Centroid of Largest Clusters : ' + str(list(map(int, cluster_centers[0]))))
plt.show()


