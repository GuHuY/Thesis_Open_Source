import numpy as np
import os

path = os.getcwd()

file_name = path+'\\2132131.txt'

data = np.loadtxt(file_name)

print(data)


