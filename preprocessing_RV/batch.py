import numpy as np
import preprocessing
from pathlib import Path

# /Users/rex/python/z_thesis/raw_ECG
input_address = '/Users/rex/python/z_thesis/raw_ECG/'
output_address = '/Users/rex/python/z_thesis/RR_data/'
# a01-a20, b01-b05, c01-c10
a = ['a'+str(x).zfill(2) for x in range(17,21)]
b = ['b'+str(x).zfill(2) for x in range(1,6)]
c = ['c'+str(x).zfill(2) for x in range(1,11)]
ECG_data_file_list = a + b + c



for file in ECG_data_file_list:
    mat_pos = input_address + file
    path = Path(output_address+file+'.txt')
    if path.exists():
        print('update ', end="")
        path.unlink()
    print(file)
    
    (result, R, A, e, f) = preprocessing.synthesize(mat_pos)
    np.savetxt(output_address + file + '.txt', result, fmt='%.5f')



