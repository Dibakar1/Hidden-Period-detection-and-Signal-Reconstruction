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

x1 = 100*signal.triang(17)
x1 = x1 - np.mean(x1)
x2 = 50*signal.triang(4)
x2 = x2 - np.mean(x2)
x3 = 25*np.random.randn(340) 
sig = np.tile(x1, 20) + np.tile(x2, 85) + np.tile(x3, 1)

x_171 = np.tile(x1, 20)
x_41 = np.tile(x2, 85)

mean_x171 = np.mean(x_171)
plt.figure(110)
plt.stem(x_171)
plt.show(x_171)
plt.figure(100)
plt.stem(x_41)
plt.show(x_41)
# sig = np.tile(50*np.random.rand(6),2) + np.tile(50*np.random.rand(4),2)
x= sig
# input signal numpy array
plt.figure(1)
plt.stem(x)
plt.show(x)
x = x
N = x.shape[0] # length of input signal
P_max = 24 # maximum period of the signal is assumed which is <= N. 

D = []
var = np.arange(1,P_max+1)
for i in var:
	if phi(i) == 1:
		D.append((phi(i))**4)
	elif phi(i) > 1:
		for j in range(phi(i)):
			D.append(phi(i)**4)

Dl = np.array(D)
D = np.diag(Dl)
x_1 = projected_x(1,x)
x_2 = projected_x(2,x)
x_4 = projected_x(4,x)
x_17 = projected_x(17,x)
x_10 = projected_x(10, x)
x_34 = projected_x(34,x)
x_85 = projected_x(85,x)
x_170 = projected_x(170,x)
x_20 = projected_x(20, x)
x_68 = projected_x(68,x)
x_340 = projected_x(340,x)
sig_hat_17 = x_17   - projected_x(17, x_340) - projected_x(17, x_34) - projected_x(17, x_68) - projected_x(17, x_170)
sig_hat_4 = x_2 + x_4 -  projected_x(2,x_340) -projected_x(2, x_34) - projected_x(2, x_10) - projected_x(4, x_20) - projected_x(2, x_170) - projected_x(4, x_68) ...
			... - projected_x(4, x_340) - projected_x(2, x_20) - projected_x(2, 68) - projected_x(2, x_68)
eng = np.zeros(341)
eng[1] = energy(x_1)
eng[4] = energy(x_4)
eng[17] = energy(x_17)
eng[340] = energy(x_340)
plt.figure(10)
plt.stem(sig_hat_4)
plt.show(sig_hat_4)
plt.figure(11)
plt.stem(sig_hat_17)
plt.show(sig_hat_17)
s = (sig_hat_17 - x_171)/x_171 
plt.figure(12)
plt.stem(s)
plt.show(s)
e_171 = (energy(sig_hat_17) - energy(x_171))/energy(x_171)
e_171
s = (sig_hat_4 - x_41)/x_41 
plt.figure(13)
plt.stem(s)
plt.show(s)
e_41 = (energy(sig_hat_4) - energy(x_41))/energy(x_41)
e_41
snr = energy(x_41+x_171)/energy(x3)
snr
