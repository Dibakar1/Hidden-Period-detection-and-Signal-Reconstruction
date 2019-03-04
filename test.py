import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal
import random
import atexit
from time import time, strftime, localtime
from datetime import timedelta

def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def log(s, elapsed=None):
    line = "="*40
    print(line)
    print(secondsToStr(), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()

def endlog():
    end = time()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))

start = time()

#########################################################################
#########################################################################
#########################################################################

def factors(N):
	lis = []
	for i in range(2,N+1):
		if N%i == 0:
			lis.append(i)
	return lis

plt.close('all')
# x1 = 100*signal.triang(7)
# x2 = 100*np.random.rand(13)
x3 = 100*signal.cosine(18)
x3 = np.tile(x3, 19*2000)
x3_1 = x3 - np.mean(x3)
x4 = 100*signal.triang(19)
x4 = np.tile(x4, 18*2000)
x4_1 = x4 - np.mean(x4)
n = 60*np.random.randn(18*19*2000)
x3 = x3_1
x4 = x4_1
n_1 = n - np.mean(n)
x =  n + x3 + x4

x = x[0:381*4]
print(np.matmul(x.T,x))
N = x.shape[0]
vari = np.zeros(int(N/2)+1)
plot_f = np.zeros(int(N/2))
plot_f1 = np.zeros(int(N/2))
snr = 10*np.log(np.matmul((x3_1+x4_1).T,x3_1+x4_1)/np.matmul(n_1.T,n_1))
print("snr:	",snr)
for i in range(2,int(N/2)+1):
	m,n = int(N/i),i
	mat_size = m*n
	data = x[0:mat_size]
	data_matrix = data.reshape(m,n)
	var = 0
	equi_class = range(n)
	for j in equi_class:
		var = var + np.var(data_matrix[:,j])
	vari[i] = var/len(equi_class)

prev = 1
current = 2
nxt = 3
max_var = np.max(vari)
plot = np.zeros(int(N/2))
for i in range(1,int(N/2)-2):
	if vari[current] < vari[prev]:
		if vari[current] < vari[nxt]:
			plot[current] = ((max_var - 3*vari[current] + vari[nxt] + vari[prev])**4)
	prev = current
	current = nxt
	nxt = nxt + 1
fp = plot
fp = fp[0:int(N/2)]
fp = fp/np.sqrt(np.matmul(fp.T,fp))
plt.figure(1)
plt.stem(fp)
plt.show()

atexit.register(endlog)
log("Start Program")