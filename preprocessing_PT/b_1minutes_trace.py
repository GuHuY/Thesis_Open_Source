import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from a_lib import *


# Get Data
DataID = 'a01'
start_time = 43*6000
end_time = start_time + 6000

ecg, sqrs125, wavedet_ = get_ecg_sqrs125_wavedet(DataID, 5)
QRS = sqrs125
features_list, baseline_list = get_feature(QRS, ecg)
features_arr = np.array(features_list)
ecg_seg = ecg[start_time: end_time]
qrs_seg = []
for val in QRS:
    if val >= end_time:
        break
    if val >= start_time:
       qrs_seg.append(val)
       
interval_list = []
for idx in range(1, len(qrs_seg)):
    interval_list.append(qrs_seg[idx] - qrs_seg[idx-1])

points = []
for idx in range(len(interval_list)-1):
    points.append([interval_list[idx], interval_list[idx+1]])
points = np.array(points)*10




plt.figure(figsize=(7,9), )
grid = plt.GridSpec(7, 4, wspace=0.5, hspace=1)
plt.subplot(grid[0:2, 0:])
plt.plot(np.linspace(start_time, end_time, 6000, endpoint=False),
         ecg_seg,
         label=DataID)
plt.plot(qrs_seg,
        [ecg[t] for t in qrs_seg],
        'x',
        label='QRS from .qrs')
# plt.plot(temp_1D_arr[:, 0],
#         temp_1D_arr[:, 1],
#         '.',
#         label='feature')
plt.xlabel('Time(0.01s)')
plt.ylabel('Voltage(v)')
plt.legend(loc='upper right')



plt.subplot(grid[2:, 0:])

# line
x = np.linspace(0,1800,1000)
plt.plot(x,x+100,c='black',label='Border')
plt.plot(x,x,c='black')
plt.plot(x,x-100,c='black')

plt.plot(points[:,0], points[:,1], label='Trace')
plt.scatter(points[:,0], points[:,1], c='r', s=5)
plt.xlabel('RRn(ms)')
plt.ylabel('RRn+1(ms)')
plt.legend(loc='upper right')
plt.xlim(0,2000)
plt.ylim(0,2000)
plt.show()



