import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal
import random
import matplotlib.pyplot as plt

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

x = np.random.rand(999)
N = x.shape[0]
p,q = 2,300
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
plt.figure(1)
plt.stem(ar,ll)
plt.show()

ll = np.array(ll)
com_period = np.argmax(ll) + p
fac = factors(com_period)
en_per = []
for i in fac:
	p = projected_x(i,x)
	e = energy(p)
	en_per.append(e/energy(x))

en_per[0] = 0
plt.figure(2)
plt.stem(fac,en_per)
plt.show()