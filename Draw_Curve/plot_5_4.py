import numpy as np 
import matplotlib.pyplot as plt


# Control_Acc=[
# [92.62],
# [90.52]
# ]

# Control_AUC=[
# [80.42],
# [69.61]
# ]

# RV_Acc=[
# [96.21],
# [90.72]
# ]

# RV_Spe=[
# [90.08],
# [84.28]
# ]

# PT_Acc=[
# [97.29],
# [97.00]
# ]

# PT_Spe=[
# [92.95],
# [91.51]
# ]

# control=[[0.8937],[0.8710]]
# RV1=[[0.8878],[0.8704]]
# PT1=[[0.9560],[0.9420]]



# plt.figure()
# plt.plot(Control_Acc, 'rx-', label='Control Acc')
# plt.plot(RV_Acc, 'ro-', label='RV(n=1) Acc')
# plt.plot(PT_Acc, 'r+-', label='PT1 Acc')
# plt.plot(Control_Spe, 'bx-', label='Control Spe')
# plt.plot(RV_Spe, 'bo-', label='RV(n=1) Spe')
# plt.plot(PT_Spe, 'b+-', label='PT1 Spe')
# scale_ls = range(0,2)
# index_ls = ['Before','After']
# plt.xlim(-0.2,1.2)
# plt.ylim(bottom=65)
# plt.xticks(scale_ls,index_ls)  ## 可以设置坐标字
# plt.legend(loc='lower left')
# plt.ylabel('Percentage(%)')
# plt.show()


control=[[0.8937],[0.8631]]
RV1=[[0.9376],[0.8860]]
# RV2=[[0.8631],[0.8805]]
PT1=[[0.9560],[0.9455]]
# PT2=[[0.8918],[0.8484]]


plt.figure()
plt.plot(control, 'ro-', label='Control')
plt.plot(RV1, 'b+-', label='RV(n=1)')
plt.plot(PT1, 'gx-', label='PT1')


scale_ls = range(0,2)
index_ls = ['No noise','SNR=20']
plt.xlim(-0.2,1.2)
plt.ylim(bottom=0.8)
plt.xticks(scale_ls,index_ls)  ## 可以设置坐标字
plt.legend(loc='lower left')
plt.ylabel('AUC')
plt.show()
