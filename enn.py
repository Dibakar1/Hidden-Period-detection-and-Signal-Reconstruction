import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal

plt.close('all')
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

def factors(N):
	lis = []
	for i in range(1,N+1):
		if N%i == 0:
			lis.append(i)
	lis = np.array(lis)
	return lis

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

def dicti(N):
	q = factors(N)
	l = []
	for i in q:
		l.append(rec_q(i))
	A = np.concatenate(l, axis = 1)
	return A

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

x3 = 100*signal.triang(7)
x3 = x3 - np.mean(x3)
x4 = 100*signal.triang(5)
x4 = x4 - np.mean(x4)
x5 = 50*np.random.randn(160)
sig = np.tile(x4, 7*5) + np.tile(x3, 5*5)
sig = sig[0:160] + x5
plt.figure(1)
plt.stem(sig)
plt.show()
cir_sig = circulant(sig)
sig = np.matmul(sig.T, cir_sig)
plt.figure(2)
plt.stem(sig)
plt.show()
x= sig
N = x.shape[0] # length of input signal

l = []
for i in range(1,80):
	f_i = projected_x(i,x)
	e_f = energy(f_i)/i
	l.append(e_f)

l = np.array(l)
plt.figure(1)
plt.stem(l/np.max(l))
plt.show()
