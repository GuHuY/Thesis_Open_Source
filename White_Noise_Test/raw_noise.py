import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from pathlib import Path
from a_lib import *

half_sample_size = 30
my_fmt = '%.4f'
for idx in range(0, half_sample_size*2):
    my_fmt += ' %.4f'


def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

SNR = 20 #db
a = ['a'+str(x).zfill(2) for x in range(1,21)]
b = ['b'+str(x).zfill(2) for x in range(1,6)]
c = ['c'+str(x).zfill(2) for x in range(1,11)]
datalist = a + b + c
for DataID in datalist:
    load_path = '/Users/rex/python/Thesis_Open_Source/RR_raw/' + DataID + '.txt'
    save_file = '/Users/rex/python/Thesis_Open_Source/White_Noise_Test/raw_Noise/' + DataID + '.txt'

    data_arr = np.loadtxt(load_path)
    data_list = list(data_arr)
    data_with_white_noist = []
    for item in data_list:
        ditry_features = list(wgn(item[:-1], SNR))
        ditry_features.append(item[-1])
        data_with_white_noist.append(ditry_features)

    path = Path(save_file)
    if path.exists():
        print('update ', end="")
        path.unlink()  # delete file
    print(DataID)
    np.savetxt(save_file, np.array(data_with_white_noist), fmt=my_fmt)
pass
