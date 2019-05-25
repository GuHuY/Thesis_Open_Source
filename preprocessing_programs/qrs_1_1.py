## detect QRS complex from ECG time series

import numpy as np 
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt

def read_ecg(file_name):
	return genfromtxt(file_name, delimiter=',')

def lgth_transform(ecg, ws):
	lgth=ecg.shape[0]
	sqr_diff=np.zeros(lgth)
	diff=np.zeros(lgth)
	ecg=np.pad(ecg, ws, 'edge')
	for i in range(lgth):
		temp=ecg[i:i+ws+ws+1]
		left=temp[ws]-temp[0]
		right=temp[ws]-temp[-1]
		diff[i]=min(left, right)
		diff[diff<0]=0
	# sqr_diff=np.multiply(diff, diff)
	# diff=ecg[:-1]-ecg[1:]
	# sqr_diff[:-1]=np.multiply(diff, diff)
	# sqr_diff[-1]=sqr_diff[-2]
	return np.multiply(diff, diff)

def integrate(ecg, ws):
	lgth=ecg.shape[0]
	integrate_ecg=np.zeros(lgth)
	ecg=np.pad(ecg, math.ceil(ws/2), mode='symmetric')
	for i in range(lgth):
		integrate_ecg[i]=np.sum(ecg[i:i+ws])/ws
	return integrate_ecg

def find_peak(data, ws):
	lgth=data.shape[0]
	true_peaks=list()
	for i in range(lgth-ws+1):
		temp=data[i:i+ws]
		if np.var(temp)<5:
			continue
		index=int((ws-1)/2)
		peak=True
		for j in range(index):
			if temp[index-j]<=temp[index-j-1] or temp[index+j]<=temp[index+j+1]:
				peak=False
				break

		if peak is True:
			true_peaks.append(int(i+(ws-1)/2))
	return np.asarray(true_peaks)

def find_R_peaks(ecg, peaks, ws):
	num_peak=peaks.shape[0]
	R_peaks=list()
	for index in range(num_peak):
		i=peaks[index]
		if i-2*ws>0 and i<ecg.shape[0]:
			temp_ecg=ecg[i-2*ws:i]
			R_peaks.append(int(np.argmax(temp_ecg)+i-2*ws))
	return np.asarray(R_peaks)

def find_S_point(ecg, R_peaks):
	num_peak=R_peaks.shape[0]
	S_point=list()
	for index in range(num_peak):
		i=R_peaks[index]
		cnt=i
		if cnt+1>=ecg.shape[0]:
			break
		while ecg[cnt]>ecg[cnt+1]:
			cnt+=1
			if cnt>=ecg.shape[0]:
				break
		S_point.append(cnt)
	return np.asarray(S_point)


def find_Q_point(ecg, R_peaks):
	num_peak=R_peaks.shape[0]
	Q_point=list()
	for index in range(num_peak):
		i=R_peaks[index]
		cnt=i
		if cnt-1<0:
			break
		while ecg[cnt]>ecg[cnt-1]:
			cnt-=1
			if cnt<0:
				break
		Q_point.append(cnt)
	return np.asarray(Q_point)

def EKG_QRS_detect(ecg, fs, QS, plot=False):
	sig_lgth=ecg.shape[0]
	ecg=ecg-np.mean(ecg)
	ecg_lgth_transform=lgth_transform(ecg, int(fs/20))
	# ecg_lgth_transform=lgth_transform(ecg_lgth_transform, int(fs/40))

	ws=int(fs/8)
	ecg_integrate=integrate(ecg_lgth_transform, ws)/ws
	ws=int(fs/6)
	ecg_integrate=integrate(ecg_integrate, ws)
	ws=int(fs/36)
	ecg_integrate=integrate(ecg_integrate, ws)
	ws=int(fs/72)
	ecg_integrate=integrate(ecg_integrate, ws)

	peaks=find_peak(ecg_integrate, int(fs/10))
	R_peaks=find_R_peaks(ecg, peaks, int(fs/40))
	if QS:
		S_point=find_S_point(ecg, R_peaks)
		Q_point=find_Q_point(ecg, R_peaks)
	else:
		S_point=None
		Q_point=None
	if plot:
		index=np.arange(sig_lgth)/fs
		fig, ax=plt.subplots()
		ax.plot(index, ecg, 'b', label='EKG')
		ax.plot(R_peaks/fs, ecg[R_peaks], 'ro', label='R peaks')
		if QS:
			ax.plot(S_point/fs, ecg[S_point], 'go', label='S')
			ax.plot(Q_point/fs, ecg[Q_point], 'yo', label='Q')
		ax.set_xlim([0, sig_lgth/fs])
		ax.set_xlabel('Time [sec]')
		ax.legend()
		# ax[1].plot(ecg_integrate)
		# ax[1].set_xlim([0, ecg_integrate.shape[0]])
		# ax[2].plot(ecg_lgth_transform)
		# ax[2].set_xlim([0, ecg_lgth_transform.shape[0]])
		plt.show()
	return R_peaks

import wfdb
from scipy.interpolate import interp1d
'''
QRS detection demo 
@author: Kemeng Chen: kemengchen@email.arizona.edu
'''

defalt_path = '/Users/rex/python/z_thesis/raw_ECG/'

def Read_ECG_and_ANN(position):
    """
    根据文件地址导入：ECG数据，注释，采样频率，记录名

    parameter:
        position(str): The position of record

    returns:
        list: ECG record
        list: Annotation ('0' stand for no apnea, '1' stand for apnea)
        int: The sampling frequency of ECG record
        str: The name of ECG record
    """
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


def Resampling(sample_value_list, original_freq, out_freq=360):
    """
    Resampling the input.

    parameters:
        B(list): The output of B()
        frequency(int): The sampling frequency apply on B

    return:
        list: frequency = 360
    """
    fill_sample_value_list = sample_value_list +[sample_value_list[-1]] * original_freq

    x_before_samp = np.linspace(0, 
                                # 输入采样点个数除以采样频率等于采样时长
                                len(fill_sample_value_list) // original_freq, 
                                len(fill_sample_value_list), 
                                endpoint=False)

    # fl为插值结果 linear cubic
    fl = interp1d(x_before_samp, fill_sample_value_list, kind='linear')
    # 采样间隔设置
    x_after_samp = np.linspace(0,
                               len(sample_value_list) // original_freq, 
                               # 采样时长乘以采样频率等于采样点数
                               len(sample_value_list) // original_freq * out_freq,
                               endpoint=False)
    # 采样
    y_after_samp = fl(x_after_samp)
    # [[int(round(x)) for x in x_after_samp], y_after_samp]
    return y_after_samp

def QRS_test(file_name):
    """
	QRS detection on file_name
	assuming 360 Hz sampling rate, may not work with very low sampling rate signal

	args:
		file_name: file containing ecg data in one column
	"""
    (ecg, annotation, samp_freq, name) = Read_ECG_and_ANN(file_name)
    resample_ecg = Resampling(ecg, samp_freq, 360) * 1000
    R_peaks = EKG_QRS_detect(resample_ecg[:100000], 360, False, True)
    print(R_peaks)

if __name__ == '__main__':
	QRS_test(defalt_path+'c07')
