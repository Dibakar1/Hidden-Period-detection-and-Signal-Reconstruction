import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal
import random
import matplotlib.pyplot as plt
import atexit
from time import time, strftime, localtime
from datetime import timedelta
from matplotlib.pyplot import *

def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def log(s, elapsed=None):
    line = "="*50
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


def factors(N):
	lis = []
	for i in range(1,N+1):
		if N%i == 0:
			lis.append(i)
	lis = np.array(lis)
	return lis

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
	G_q = basis[:,0:phi(q)]
	return G_q

def projector(q):
	r = rec_q(q)
	p = np.linalg.pinv(np.matmul(r.T,r))
	p = np.matmul(p,r.T)
	P_q = np.matmul(r,p)
	P_q = P_q/q
	return P_q

def projected_x(q, x):
	xq_i = np.matmul(projector(q),x)
	norm = np.matmul(xq_i.T, xq_i)
	xq_i = xq_i/np.sqrt(norm)
	alpha = np.matmul(xq_i.T, x)
	proj = alpha*xq_i
	return proj

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
listt = []
snrr = []
oo = np.arange(10,30,10)
for i in oo:
	x5 = i*np.random.rand(8*16*11*3)
	sig = x3 + x2 + x5 + x1
	sig = sig[0:6600]
	# plt.figure(11)
	# plt.stem(sig)
	# plt.show()
	x = sig
	N = x.shape[0]
	snr = 10*np.log(energy(x3 + x1 + x2)/energy(x5))
	print('i-th value snr in dB:',i, snr)
	p,q = 5,1000 
	ll = []
	for i in range(p,q+1):
		m,n = int(N/i), i
		ss = m*n
		sig = x[0:ss]
		data_matrix = sig.reshape(m,n)
		u,s,vh = np.linalg.svd(data_matrix)
		si = s[0]/s[1]
		ll.append(si)

	ar = np.arange(p, q+1)
	# plt.figure(1)
	# title('SVD of the data matrix formed')
	# xlabel('Assumed Period (P)')
	# ylabel('$\lambda_1 / \lambda_2$')
	# plt.stem(ar,ll)
	# plt.show()

	ll = np.array(ll)
	com_period = np.argmax(ll) + p
	fac = factors(com_period)
	en_per = []
	for i in fac:
		p = projected_x(i,x)
		e = energy(p)
		en_per.append(e)

	en_per[0] = 0
	en_every = en_per/energy(x)
	plt.figure(2)
	title('Individual component strengths')
	xlabel('Hidden Periods ($p_i$)')
	ylabel('Normalized Strengths')
	plt.stem(fac,en_every)
	plt.show()
	sor = np.sort(en_every)
	listt.append(sor[-4])
	snrr.append(snr)

	sig8 = projected_x(8,x)
	# plt.figure(3)
	# plt.stem(sig8)
	# plt.show()

	sig11 = projected_x(11,x)
	# plt.figure(4)
	# plt.stem(sig11)
	# plt.show()

	sig16 = projected_x(16,x)
	# plt.figure(5)
	# plt.stem(sig16)
	# plt.show()

plt.figure(1)
title('Highest noisy component Vs. SNR plot')
xlabel('SNR')
ylabel('Strength of highest noisy component')
plt.stem(snrr,listt)
plt.show()
atexit.register(endlog)
log("Start Program")