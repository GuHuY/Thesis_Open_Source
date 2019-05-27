import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import wfdb
import pywt   # python 小波变换的包

# 取数据
position = 'apnea_ECG_data/a02'


def read_data(position):
    temp = wfdb.rdrecord(position)
    ecg_data = temp.p_signal
    samp_freq = temp.fs
    reco_name = temp.record_name
    temp = wfdb.rdann(position, 'apn')
    anno_data = temp.symbol
    converter = {'N': 0, 'A': 1}
    return ([x[0] for x in ecg_data],
            [converter[x] for x in anno_data if x in ['N', 'A']],
            samp_freq,
            reco_name)


(ecg, annotation, sf, name) = read_data(position)
ecg = np.array(ecg)

wavefunc = pywt.Wavelet('db1')
coeff = pywt.wavedec(ecg, wavefunc, mode='sym', level=4)
