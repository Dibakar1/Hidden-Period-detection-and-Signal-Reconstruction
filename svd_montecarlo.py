import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal
import random
import time

x3 = 100*signal.triang(8)
x3 = np.tile(x3, 16*11*10)
x3_1 = x3 - np.mean(x3)
x1 = 80*signal.cosine(11)
x1 = np.tile(x1, 8*16*10)
x1_1 = x1 - np.mean(x1)
x2 = 70*signal.triang(16)
x2 = np.tile(x2, 8*11*10)
x2_1 = x2 - np.mean(x2)

l = []

for length in range(900, 11000, 506):
	cum_dib = 0
	start = time.time()
	for sing_itr in range(2):
		oo = np.arange(20,21,2)
		for i in oo:
			x5 = i*np.random.rand(8*16*11*10)
			sig = x3 + x2 + x5 + x1
			sig = sig[0:length]
			x = sig
			N = x.shape[0]

			p,q = 2, int(length/2)
			ll = []
			for i in range(p,q+1):
				m,n = int(N/i), i
				ss = m*n
				sig = x[0:ss]
				data_matrix = sig.reshape(m,n)
				u,s,vh = np.linalg.svd(data_matrix)
				si = s[0]/s[1]
				ll.append(si)

			# ar = np.arange(p, q+1)
			# plt.figure(1)
			# plt.stem(ar,ll)
			# plt.show()

		end = time.time()
		dib = end - start
		cum_dib = cum_dib + dib

	l.append(cum_dib/2)

print(l)
plt.figure(100)
plt.stem(np.arange(len(l)), l)
plt.show()
