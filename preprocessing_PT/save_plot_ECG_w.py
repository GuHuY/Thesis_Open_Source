import numpy as np
import matplotlib.pyplot as plt
from a_lib import *



a = ['a'+str(x).zfill(2) for x in range(1,21)]
b = ['b'+str(x).zfill(2) for x in range(1,6)]
c = ['c'+str(x).zfill(2) for x in range(1,11)]
datalist = a + b + c
for DataID in datalist:
    ecg, sqrs125, wavedet_ = get_ecg_sqrs125_wavedet(DataID, 5)
    QRS = sqrs125
    #----------     save plot    ----------#
    plt.cla()
    plt.plot(ecg, label=DataID)
    plt.plot(QRS,
            [ecg[t] for t in QRS],
            'x',
            label='QRS from .qrs')
    plt.xlabel('Time(0.01s)')
    plt.ylabel('Voltage(v)')
    plt.legend(loc='upper right')
    plt.savefig("plot_simple_ECG_w/" + DataID + ".png")
    #----------------------------------------#
    pass