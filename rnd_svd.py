import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import time


x3 = 100*np.random.randn(9)
# plt.figure(100)
# plt.stem(x3)
# plt.show()
x3 = np.tile(x3, int(12321*5/9))
# x3 = x3 - np.mean(x3)
x1 = 90*np.random.randn(111)
# plt.figure(101)
# plt.stem(x1)
# plt.show()
x1 = np.tile(x1, int(12321*5/111))
# x1 = x1 - np.mean(x1)
x4 = 95*np.random.randn(37)
x4 = np.tile(x4, int(12321*5/37))
# # plt.figure(102)
# plt.stem(x2)
# plt.show()
x2 = 100*np.random.randn(12321)
x2 = np.tile(x2,5)
# x2 = x2 - np.mean(x2)
listt = []
snrr = []
length = []
time_elasped = []
low_len, high_len = 400, 700
gap = 50
trnk = np.arange(low_len, high_len+1, gap)
for j in trnk:
	start = time.time()
	oo = np.arange(220,222,2)
	for k in oo:
		x5 = k*np.random.rand(12321*5)
		sig = x1 + x2 + x3 + x4 + x5 
		#sig = sig[0:50000]
		# plt.figure(11)
		# plt.stem(sig[0:390])
		# plt.show()
		x = sig
		N = x.shape[0]
		snr = 10*np.log(np.matmul((x1 + x2 + x3 + x4).T,(x1 + x2 + x3 +x4))/np.matmul(x5.T,x5))
		print('i-th value snr in dB:',k, snr)
		p,q = 5, int(N/2)
		ll = []
		for i in range(p,q+1):
			m,n = int(N/i), i
			ss = m*n
			sig = x[0:ss]
			data_matrix = sig.reshape(m,n)
			if n<j:
				data_matrix = data_matrix[0:int(n/2),0:int(n)]
			else:
				data_matrix = data_matrix[0:int(j/2),0:int(j)]
			# # print(data_matrix.shape)
			u,s,vh = np.linalg.svd(data_matrix)
			si = s[0]/s[1]
			ll.append(si)

		ar = np.arange(p, q+1)
		# plt.figure(1)
		# title('Truncated DataMatrix SVD')
		# xlabel('Period')
		# ylabel('sigma1/sigma2')
		# plt.stem(ar,ll)
		# plt.show()
		tres = np.argmax(ll) + p
		if tres>int(N/4):
			tres = 0
		else:
			tres = tres
		listt.append(tres)

	length.append(j)
	end = time.time()
	taken = end - start
	time_elasped.append(taken)

plt.figure(2)
title('highest peak & length')
xlabel('Truncated Length Taken')
ylabel('Detected highest peak (Detected Period)')
plt.stem(length,listt)
plt.show()
# print(time_elasped)
plt.figure(3)
title('Time Elasped & Length')
xlabel('Length')
ylabel('Time Elasped in sec')
plt.stem(length, time_elasped)
plt.show()
print(time_elasped)
set_len = set(listt)
print(set_len)