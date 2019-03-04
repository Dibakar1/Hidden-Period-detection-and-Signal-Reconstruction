import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.pyplot import *

def phi(n):
	if n == 0:
		return 0
	num = 0
	for k in range(1, n+1):
		if f.gcd(n,k) == 1:
			num = num+1
	return num

def c(q):
	k = []
	for i in range(q):
		if f.gcd(i,q) == 1:
			k.append(i)
	k = np.array(k)
	c = []
	for n in range(q):
		p = np.sum(np.cos(2*np.pi*k*n/q))
		c.append(p)
	# c = np.array(c)
	return c

def rec_q(q):
	if (len(c(q))<N):
		div = int(N/len(c(q)))
		quo = N%len(c(q))
		vf = c(q)
		vff = div*vf
		full = vff + vf[0:quo]
	if (len(c(q))==N):
		full = c(q)
	full = np.array(full)
	basis = circulant(full)
	basis = basis[:,0:phi(q)]
	return basis

def dicti(P_max):
	q = np.arange(1,P_max+1)
	l = []
	for i in range(P_max):
		l.append(rec_q(q[i]))
	A = np.concatenate(l, axis = 1)
	return A

def proj_q(q):
	if (len(c(q))<N):
		div = int(N/len(c(q)))
		quo = N%len(c(q))
		vf = c(q)
		vff = div*vf
		full = vff + vf[0:quo]
	if (len(c(q))==N):
		full = c(q)
	full = np.array(full)
	basis = circulant(full)
	return basis

def smooth(x,window_len=3,window='hanning'):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def energy(x):
	eng = np.matmul(x.T, x)
	return(eng)

x3 = 100*signal.triang(8)
x3 = np.tile(x3, 16*11*3)
x3 = x3 - np.mean(x3)
x1 = 45*signal.cosine(11)
x1 = np.tile(x1, 8*16*3)
x1 = x1 - np.mean(x1)
x2 = 50*signal.triang(16)
x2 = np.tile(x2, 8*11*3)
x2 = x2 - np.mean(x2)
x5 = 15*np.random.randn(8*16*11*3)
trnc = 180
sig = x1 + x2 + x3
snr = 10*np.log(energy(sig[0:trnc])/energy(x5[0:trnc]))
print('snr:', snr)
sig = sig[0:trnc] + x5[0:trnc] 
plt.figure(11)
plt.stem(sig)
plt.show()
cir_sig = circulant(sig)
sig = np.matmul(sig.T, cir_sig)
# sig = smooth(sig)
x= sig
N = x.shape[0] # length of input signal
P_max = N # maximum period of the signal is assumed which is <= N. 
D = []
var = np.arange(1,P_max+1)
for i in var:
	if phi(i) == 1:
		D.append((phi(i))**2)
	elif phi(i) > 1:
		for j in range(phi(i)):
			D.append(phi(i)**2)
#
Dl = np.array(D)
D = np.diag(Dl)
a = dicti(P_max)
B = np.matmul(a,np.linalg.pinv(D))
A_hat = B
mul = np.matmul(A_hat, A_hat.T)
mul = np.linalg.pinv(mul)
mu = np.matmul(A_hat.T, mul)
y = np.matmul(mu, x)
y[0] = 0
q_old = 0
l = []
for i in var:
	q_i = phi(i)
	x = np.sum(np.square(np.abs(y[q_old:q_i + q_old])))
	l.append(x)
	q_old = q_i + q_old
#
plt.figure(3)
title('Ramanujan Process hidden period finding')
xlabel('Hidden periods $p_i$')
ylabel('Strength of the hidden components')
plt.stem(np.arange(trnc),l)
plt.show()