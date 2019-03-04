import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal
import random
import time
from matplotlib.pyplot import *


def factors(N):
	lis = []
	for i in range(2,N):
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

plt.close('all')
x3 = 100*signal.triang(8)
x3 = np.tile(x3, 17*11*10)
x3_1 = x3 - np.mean(x3)
x1 = 80*signal.cosine(11)
x1 = np.tile(x1, 8*17*10)
x1_1 = x1 - np.mean(x1)
x2 = 70*signal.triang(17)
x2 = np.tile(x2, 8*11*10)
x2_1 = x2 - np.mean(x2)
l = []
low_len, high_len = 900, 11000
stride = 506
for length in range(low_len, high_len, stride):
	cum_dib = 0
	start = time.time()
	for sing_itr in range(5):
		plotnew = np.zeros(int(length/2))
		plotnew_2 = np.zeros(int(length/2))
		resends = 7
		eqi_cls = 9
		start = time.time()
		for t in range(resends):
			n = 20*np.random.randn(11*17*8*10)
			n_1 = n - np.mean(n)
			x =  n + x1 + x2 + x3

			x = x[0:length]
			#print(np.matmul(x.T,x))
			N = x.shape[0]
			vari = np.zeros(N)
			#snr = 10*np.log(np.matmul((x3_1+x2_1+x1_1).T,x3_1+x1_1+x2_1)/np.matmul(n_1.T,n_1))
			#print("snr:	",snr)
			itr = 1
			for p in range(itr):
				for i in range(2,int(N/2)+1):
					m,n = int(N/i),i
					mat_size = m*n
					data = x[0:mat_size]
					data_matrix = data.reshape(m,n)
					var = 0
					if n < eqi_cls+1:
						equi_class = range(n)
					else:
						equi_class = range(10)
					for j in equi_class:
						var = var + np.var(data_matrix[0:eqi_cls,j])
					vari[i] = var/len(equi_class) - np.var(x[0:eqi_cls])
					# plt.figure(11)
					# title('Variance Figure is Plotted')
					# xlabel('Assumed Period ($P$)')
					# ylabel('Variance')
					# plt.stem(vari)
					# plt.show()

				prev = 1
				current = 2
				nxt = 3
				plot2 = np.zeros(int(N/2))
				plot1 = np.zeros(int(N/2))
				max_var = np.max(vari)
				for i in range(1,int(N/2)-2):
					if vari[current] < vari[prev]:
						if vari[current] < vari[nxt]:
							plot1[current] = ((max_var - vari[current])**4)
							plot2[current] = 1
					prev = current
					current = nxt
					nxt = nxt + 1
				
				for idnx,val in enumerate(plot1):
					if idnx < N/2:
						pass
					else:
						if ((plot1[idnx] is not 0) and (plot1[int(0.5*idnx)] is 0)):
							plot1[idnx] = 0
				plotnew = plotnew + plot2
				plotnew_2 = plotnew_2 + plot1

		for i in range(plotnew.shape[0]):
			if plotnew[i] >= np.median(plotnew):
				plotnew[i] = plotnew[i]**4
			#else:
			#	plotnew[i] = 0

		fp = plotnew[0:int(N/4)]*plotnew_2[0:int(N/4)]
		fp = fp/np.sqrt(np.matmul(fp.T,fp))

		# plt.figure()
		# title('Variance vs. Assumed Period')
		# xlabel('Assumed Period')
		# ylabel('Strength of Dip')
		# plt.stem(fp)
		# plt.show()

		end = time.time()
		dib = end - start
		cum_dib = cum_dib + dib

		# print(np.argmax(fp)%176)
		# print('buddy')
		# plt.figure(10)
		# plt.stem(fp)
		# plt.show()
		ffp = np.argmax(fp)
		if (ffp%176) == 0:
			cum_dib = cum_dib
		else:
			cum_dib = -1


	l.append(cum_dib/5)

print(l)
plt.figure(100)
title('Monte-Carlo Simulation')
xlabel('Length of the signal')
ylabel('Time in sec')
plt.stem(np.arange(low_len, high_len, stride),l)
plt.show()