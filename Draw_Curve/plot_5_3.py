import numpy as np 
import matplotlib.pyplot as plt


# Control_Acc=[
# [92.62],
# [90.42]
# ]

# Control_Spe=[
# [80.42],
# [82.11]
# ]

# RV_Acc=[
# [96.21],
# [94.44]
# ]

# RV_Spe=[
# [90.08],
# [75.94]
# ]

# PT_Acc=[
# [97.29],
# [97.00]
# ]

# PT_Spe=[
# [92.95],
# [90.12]
# ]

control=[[0.8937],[0.8710]]
RV1=[[0.9376],[0.8704]]
RV3=[[0.8878],[0.8805]]
PT1=[[0.9560],[0.9420]]
PT2=[[0.8918],[0.8484]]


plt.figure()
plt.plot(control, 'ro-', label='Control')
plt.plot(RV1, 'bx-', label='RV(n=1)')
plt.plot(RV3, 'b+-', label='RV(n=3)')
plt.plot(PT1, 'gx-.', label='PT1')
plt.plot(PT2, 'g+-.', label='PT2')

scale_ls = range(0,2)
index_ls = ['sqrs125','wavedet']
plt.xlim(-0.2,1.2)
plt.ylim(bottom=0.8)
plt.xticks(scale_ls,index_ls)  ## 可以设置坐标字
plt.legend(loc='lower left')
plt.ylabel('AUC')
plt.show()
