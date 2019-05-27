import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from a_lib import *


# Get Data
DataID = 'a09'
ecg, sqrs125, wavedet_ = get_ecg_sqrs125_wavedet(DataID, 5)
QRS = wavedet_
features_list, baseline_list = get_feature(QRS, ecg)
features_arr = np.array(features_list)

def get_relative_height(moment):
    segment = ecg[moment-16: moment-6] + ecg[moment+6: moment+16]
    base_mean = (sum(segment)-max(segment)-min(segment)) / len(segment)
    base_median = median(segment)
    base_line_height = (base_median+base_mean)/2
    relative_height = ecg[moment] - base_line_height
    return relative_height

# Step 1
# Get H_mean and (X, Y)
H_mean = get_mean_height(features_list)
bandwidth = estimate_bandwidth(features_arr[:,1:3], quantile=0.5, n_samples=1000)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(features_arr[:,1:3])
centroid = ms.cluster_centers_[0]
normal_interval = (centroid[0]+centroid[1])/2


# Step 2
# Outlier detection
QRS_too_high_idx_list = []
for idx, val in enumerate(features_list):
    if abs(val[0]) > 3*H_mean:
        QRS_too_high_idx_list.append(idx)
    elif val[1]+val[2] < normal_interval/2:
        QRS_too_high_idx_list.append(idx)
features_arr = np.delete(features_arr, QRS_too_high_idx_list, axis=0)
features_list = list(features_arr)
H_mean = get_mean_height(features_list)

# Step 3'
# Error detection
# 找到对于轨迹特征高度分布不在主集群中的点
# 首先确定主集群的高度
# for idx in QRS_too_high_idx_list:
#     if feature = 10 then 
trace_list = get_trace(features_list, H_mean)
#算出坐标
X = []
Y = []
Z = []
for item in trace_list:
    for posi in item[1]:
        X.append(posi[0])
        Y.append(posi[1])
        Z.append(item[3])
X = np.array(X)*10
Y = np.array(Y)*10
Z = np.array(Z)
fig = plt.figure()
ax = Axes3D(fig)
m_sactter = ax.scatter(X, Y, Z, c=Z, cmap=plt.get_cmap('rainbow'), s=1)
fig.colorbar(m_sactter, shrink=0.5)
ax.set_xlim(0, 1.1*X.max())
ax.set_ylim(0, 1.1*Y.max())
ax.set_zlim(1.1*Z.min(), 1.1*Z.max())
# ax.plot(Y, Z, 'k.', zdir='x', zs=1.1*X.max())
# ax.plot(X, Z, 'k.', zdir='y', zs=1.1*Y.max())
# ax.plot(X, Y, 'k.', zdir='z', zs=1.1*Z.min())
# ax.legend(loc='best')
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.view_init(90, 270)
# ax.view_init(30, 270-45)
# ax.view_init(30, 45)
plt.show()
pass

###### 统计基线到QRS高度差 找到切割点 ######
bin_num, bin_width, centre = 221 , 0.1 , 0
hist_bias = bin_num / 2 * bin_width
lins_bias = (bin_num-1) / 2 * bin_width
hist, bin_left_edges = np.histogram(Z,
                                    bins=bin_num,
                                    range=(centre-hist_bias, centre+hist_bias),
                                    density=True)
hist = hist/sum(hist) * 100
bin_centers = bin_left_edges[:-1] + bin_width / 2
statistics = interp1d(bin_centers, hist, kind='linear')
statistics_x = np.linspace(-10, 10, 2000, endpoint=False)
statistics_y = statistics(statistics_x)
plt.cla()
plt.figure()
plt.plot(statistics_x, statistics_y)
plt.show()


max_hist_idx = np.where(hist==np.max(hist))[0][0]
former_hist = np.max(hist)
for x in range(max_hist_idx, max_hist_idx+60):
    curr_hist = hist[x]
    if curr_hist < 1:
        right_cut = x
        break
    elif curr_hist > former_hist:
        right_cut = x
        break
    else:
        former_hist = curr_hist
right_cut = (right_cut - (bin_num+1)/2)*bin_width
former_hist = np.max(hist)
for x in range(max_hist_idx, max_hist_idx-60, -1):
    curr_hist = hist[x]
    if curr_hist < 1:
        left_cut = x
        break
    elif curr_hist > former_hist:
        left_cut = x
        break
    else:
        former_hist = curr_hist
left_cut = (left_cut - (bin_num-1)/2)*bin_width
print(left_cut)
print(right_cut)

QRS_error_idx_list = []
for idx, val in enumerate(features_list):
    trace_height = 
    if val[0] < 3*H_mean:
        QRS_too_high_idx_list.append(idx)
    elif val[1]+val[2] < normal_interval/2:
        QRS_too_high_idx_list.append(idx)
features_arr = np.delete(features_arr, QRS_too_high_idx_list, axis=0)
features_list = list(features_arr)
H_mean = get_mean_height(features_list)





# Step 3
# pca = PCA(n_components=1, whiten=True)
# pca.fit(features_arr_after_step_2)
# feature_1D = pca.transform(features_arr_after_step_2)
# model_1D = pca.transform(np.array([[H_mean, centroid[0], centroid[1]]]))
# print('explained_variance_ratio_:', pca.explained_variance_ratio_)
# print('components_:', pca.components_)
# model_1D = 3
# print(model_1D)





# Step 4
# trace_list = get_trace(list(features_arr_after_step_2), H_mean)
# candidates = []
# for val in trace_list:
#     candidates.append([val[3], val[1][0][0], val[1][0][1], val[0]+1])
#     if val[2] > 1:
#         candidates.append([val[3], val[1][1][0], val[1][1][1], val[0]+2])
# candidates_arr = np.array(candidates)[:, :3]

# temp_1D_list = []
# for idx in range(len(candidates)):
#     item = candidates[idx]
#     QRS_idx = item[3]
#     start = QRS[QRS_idx-1]
#     end = QRS[QRS_idx+1]
#     interval_20percent = int((end - start)/5)
#     RR_im1 = start-QRS[QRS_idx-2]
#     for moment in range(start+interval_20percent, min(start+int(2*normal_interval), end-interval_20percent)):
#         curr_H = abs(ecg[moment] - get_local_baseline(moment))/H_mean
#         RR_i = moment - start
#         curr_1D = curr_H * (RR_i * (2*normal_interval-RR_i))/(normal_interval**2)
#         # print(curr_1D-model_1D)
#         temp_1D_list.append([moment, curr_1D])

        


# temp_1D_arr = np.array(temp_1D_list)

# plt.figure()
# plt.plot(ecg, label=DataID)
# plt.plot(QRS,
#         [ecg[t] for t in QRS],
#         'x',
#         label='QRS from .qrs')
# plt.plot(temp_1D_arr[:, 0],
#         temp_1D_arr[:, 1],
#         '.',
#         label='feature')
# plt.xlabel('Time(0.01s)')
# plt.ylabel('Voltage(v)')
# plt.legend(loc='upper right')
# plt.show()





