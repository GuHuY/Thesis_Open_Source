import numpy as np 


# /Users/rex/python/z_thesis/raw_ECG

# RR_data_address = '/Users/rex/python/z_thesis/RR_trace_less/'
RR_data_address = '/Users/rex/python/Thesis_Open_Source/RR_left_wavedet/'
Output_address = RR_data_address
# a01-a20, b01-b05, c01-c10
a = ['a'+str(x).zfill(2) for x in range(1,21)]
b = ['b'+str(x).zfill(2) for x in range(1,6)]
c = ['c'+str(x).zfill(2) for x in range(1,11)]
file_list = a + b + c
# list_exception = ['b04', 'b05', 'c01', 'c02','c04','c09','c10']
# file_list = [i for i in file_list if i not in list_exception]

# combined_data = []
# for file in file_list:
#     file_name = RR_data_address + file + '.txt'
#     data = np.loadtxt(file_name).tolist()
#     combined_data.append(data)

# np.savetxt(Output_address +'combine.txt', 
#            np.array(combined_data),
#            fmt='%s')


fully_combined_data = []
for file in file_list:
    file_name = RR_data_address + file + '.txt'
    data = np.loadtxt(file_name).tolist()
    fully_combined_data.extend(data)

# np.savetxt(Output_address +'trace_sqrs_combine.txt', 
#            np.array(fully_combined_data),
#            fmt='%d')


np.savetxt(Output_address +'left_wavedet_combine.txt', 
           np.array(fully_combined_data),
           fmt='%.4f')