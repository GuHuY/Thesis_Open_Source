import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from a_lib import *


# Get Data
DataID = 'a01'

ecg, sqrs125, wavedet_ = get_ecg_sqrs125_wavedet(DataID, 5)
QRS = wavedet_
features_list, _ = get_feature(QRS, ecg)

#根据IntervalSum去除时间异常,根据Height去除高度异常
outlier_list = []
for idx, val in enumerate(features_list):
    outlier_list.append([sum(val), val[0], val[1:]])
# IntervalSum 排序
outlier_list.sort()
margin = int(len(outlier_list)/1000)
outlier_list = outlier_list[margin: -margin]
# Height 排序
outlier_list.sort(key=lambda x:x[1])
margin = int(len(outlier_list)/1000)
outlier_list = outlier_list[margin: -margin]

new_features_list = []
for val in outlier_list:
    new_features_list.append([val[1], val[2][0], val[2][1]])

#


#算出坐标
new_features_list = np.array(new_features_list)
X = new_features_list[:,1]*10
Y = new_features_list[:,2]*10
Z = new_features_list[:,0]


# 绘制散点图
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