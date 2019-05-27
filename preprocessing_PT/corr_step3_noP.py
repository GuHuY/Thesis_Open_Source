import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from a_lib import *


ecg, sqrs125, wavedet_ = get_ecg_sqrs125_wavedet('c03', 5)
features_list, baseline_list = get_feature(sqrs125, ecg)
features_arr = np.array(features_list)

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
print(cluster_center, RR_mean, RR_Border_Low ,RR_Border_High)

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

X = features_arr[:,1]*10
Y = features_arr[:,2]*10

#-----------------观察拒绝域效果----------------------#
# plt.figure()
# plt.scatter(X, Y, marker='.', alpha=0.3, s=5)
# plt.xlabel('RRn(ms)')
# plt.ylabel('RRn+1(ms)')
# plt.show()
#----------------------------------------------#


# Z = features_arr[:,0]
#修正H为H'
Z = []
temp = list(features_arr[:, 0])
temp = [temp[0]] + temp + [temp[-1]]
for idx in range(1, len(temp)-1):
    Z.append(min([temp[idx-1], temp[idx], temp[idx+1]]))
Z = np.array(Z)

#--------------观察三维散点分布-------------------#
# fig = plt.figure()
# ax = Axes3D(fig)
# m_sactter = ax.scatter(X, Y, Z, c=Z, cmap=plt.get_cmap('rainbow'), s=1)
# fig.colorbar(m_sactter, shrink=0.5)
# ax.set_xlim(0, 1.1*X.max())
# ax.set_ylim(0, 1.1*Y.max())
# ax.set_zlim(1.1*Z.min(), 1.1*Z.max())
# ax.set_zlabel('Z')
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# # ax.view_init(90, 270)
# # ax.view_init(30, 270-45)
# # ax.view_init(30, 45)
# ax.view_init(30, 250)
# plt.show()
#----------------------------------------------#

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
print('H_border:', H_border)


#-----------------观察拒绝域效果----------------------#
X = features_arr[:,1]*10
Y = features_arr[:,2]*10
plt.figure()
plt.scatter(X, Y, marker='.', alpha=0.3, s=5)
plt.xlabel('RRn(ms)')
plt.ylabel('RRn+1(ms)')
plt.show()
#----------------------------------------------#




pass
