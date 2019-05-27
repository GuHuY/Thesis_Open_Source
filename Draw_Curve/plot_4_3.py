import numpy as np 
import matplotlib.pyplot as plt


Control_Acc=[
[92.62],
[90.42]
]

Control_Spe=[
[80.42],
[82.11]
]

RV_Acc=[
[96.21],
[94.44]
]

RV_Spe=[
[90.08],
[75.94]
]

PT_Acc=[
[97.29],
[97.00]
]

PT_Spe=[
[92.95],
[90.12]
]

plt.figure()
plt.plot(Control_Acc, 'rx-', label='Control Acc')
plt.plot(RV_Acc, 'ro-', label='RV(n=1) Acc')
plt.plot(PT_Acc, 'r+-', label='PT1 Acc')
plt.plot(Control_Spe, 'bx-', label='Control Spe')
plt.plot(RV_Spe, 'bo-', label='RV(n=1) Spe')
plt.plot(PT_Spe, 'b+-', label='PT1 Spe')
scale_ls = range(0,2)
index_ls = ['sqrs125','wavedet']
plt.xlim(-0.2,1.2)
plt.ylim(bottom=65)
plt.xticks(scale_ls,index_ls)  ## 可以设置坐标字
plt.legend(loc='lower left')
plt.ylabel('Percentage(%)')
plt.show()
