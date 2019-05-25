import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from pathlib import Path
from a_lib import *

half_sample_size = 30
my_fmt = '%d'
for idx in range(0, half_sample_size*2):
    my_fmt += ' %d'

a = ['a'+str(x).zfill(2) for x in range(1,21)]
b = ['b'+str(x).zfill(2) for x in range(1,6)]
c = ['c'+str(x).zfill(2) for x in range(1,11)]
datalist = a + b + c
for DataID in datalist:
    ecg, wavedet_, annotation, samp_freq = get_ecg_wavedwet_anno_sf(DataID, 5)
    QRS = wavedet_
    # ecg, sqrs125, annotation, samp_freq = get_ecg_wavedwet_anno_sf(DataID, 5)
    # QRS = sqrs125
   
    T = samp_freq * 60
    QRS_list = list(QRS_arr)
    QL_len = len(QRS_list)
    wedge = 0
    sample_with_label = []
    for idx, label in enumerate(annotation):
        middle_Time = int(idx * T + T/2)
        for i in range(wedge, QL_len):
            if QRS_list[i] > middle_Time:
                middle_QRS_idx = wedge = i
                break
        sample_list = list(Indicator[middle_QRS_idx-half_sample_size:middle_QRS_idx+half_sample_size])
        sample_list.append(label)
        if len(sample_list) == half_sample_size*2+1:
            sample_with_label.append(sample_list)
    np.savetxt(file_path, np.array(sample_with_label), fmt=my_fmt)
pass
