import numpy as np

a = np.random.randint(-100, 100, size=(100000,2))
print(a)
data_list = []
for val in a:
    data_list.append([val[0], val[1], val[0]/val[1]])
output = np.array(data_list)
np.save("DivData.npy",output)
print(output)

a = np.random.randint(-100, 100, size=(100000,2))
print(a)
data_list = []
for val in a:
    data_list.append([val[0],val[0],val[1], val[1], val[0]*val[1]])
output = np.array(data_list)
np.save("MulData.npy",output)
print(output)

a = np.random.randint(-100, 100, size=(100000,2))
print(a)
data_list = []
for val in a:
    data_list.append([val[0], val[1], 78*val[0]+3*val[1]])
output = np.array(data_list)
np.save("LinearData.npy",output)
print(output)

# a = np.random.randint(-100, 100, size=(1000,3))
# print(a)
# np.save("MulData.npy",a)

