# Check the result of wavedet

import numpy as np
import matplotlib.pyplot as plt
from a_lib import *


DataID = 'a01'
ecg, sqrs125, wavedet_ = get_ecg_sqrs125_wavedet(DataID, 5)
QRS = wavedet_
#----------     save plot    ----------#
plt.figure()
plt.plot(ecg, label=DataID)
plt.plot(QRS,
        [ecg[t] for t in QRS],
        'x',
        label='QRS from .qrs')
plt.xlabel('Time(0.01s)')
plt.ylabel('Voltage(v)')
plt.legend(loc='upper right')
plt.show()
#----------------------------------------#
