# Check the result of wavedet

import numpy as np
import matplotlib.pyplot as plt

equal_speed_line = np.linspace(0,2000,1000)

N = np.random.normal(size=(100, 2))
normal_RR = N * 100 + 900

outlier = np.array([[600, 1400], [1400, 1300], [1300, 1200], [1200, 1100], [1100, 1000]])

plt.figure()
plt.plot(normal_RR[:,0], normal_RR[:,1], '.',label='normal RR')
plt.plot(outlier[:,0], outlier[:,1], 'x', label='abnormal RR')
plt.plot(equal_speed_line, equal_speed_line, label='y=x')
plt.xlim(250, 1750)
plt.ylim(250, 1750)
plt.xlabel('Time(ms)')
plt.ylabel('Time(ms)')
plt.legend(loc='upper right')
plt.show()
pass