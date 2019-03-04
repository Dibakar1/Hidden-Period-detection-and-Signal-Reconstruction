import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal
import random

def factors(N):
	lis = []
	for i in range(1,N+1):
		if N%i == 0:
			lis.append(i)
	lis = np.array(lis)
	return lis

plt.close('all')
# x1 = 100*signal.triang(7)
# x2 = 100*np.random.rand(13)
x3 = 100*signal.cosine(14)
x3 = x3 - np.mean(x3)
x4 = 100*signal.triang(19)
x4 = x4 - np.mean(x4)
n = 120*np.random.rand(7*19*10)
x3 = np.tile(x3, 19*5)
x4 = np.tile(x4, 14*5)
x =  n + x3 + x4

x = x[0:1300]
print(np.matmul(x.T,x))
N = x.shape[0]
vari = np.zeros(N)
plotnew = np.zeros(int(N/2))
snr = 10*np.log(np.matmul((x3+x4).T,x3+x4)/np.matmul(n.T,n))
print("snr:	",snr)
for p in range(10):
	for k in range(2,int(N/2)):
		#print("\n***************************\n")
		#print("\nTesting if period is ", k)
		cumm = 0
		if(k > 30):
			equi_class = random.sample(range(0, k), 30)
		else:
			equi_class = range(k)
		for i  in equi_class:
			collection = []
			if(int(N/k) > 30):
				randz = random.sample(range(0, int(N/k)), 30)
			else: 
				randz = range(int(N/k)) 
			for j in randz :
				collection.append(x[k*j+i])
				var = np.var(collection)
			cumm = cumm + 2*var - 0.5*var/(len(randz))#######################################################3
		vari[k] = cumm/(len(equi_class))
		#print("For k = ",k , "distance = ", cumm)
		#print("**********************************\n")
	#plt.figure(1)
	#plt.stem(vari)
	#plt.show()
	prev = 1
	current = 2
	nxt = 3
	plot = np.zeros(int(N/2))
	max_var = np.max(vari)
	for i in range(1,int(N/2)-2):
		if vari[current] < vari[prev]:
			if vari[current] < vari[nxt]:
				plot[current] = ((max_var - vari[current])**4)
		prev = current
		current = nxt
		nxt = nxt + 1
	for idnx,val in enumerate(plot):
		if idnx < N/2:
			pass
		else:
			if ((plot[idnx] is not 0) and (plot[int(0.5*idnx)] is 0)):
				plot[idnx] = 0
	plotnew = plotnew + plot
plotnew = plotnew/np.sqrt(np.matmul(plotnew.T,plotnew))
maxi = np.max(plotnew)
for i in range(plotnew.shape[0]):
	if(plotnew[i] < maxi/2):
		plotnew[i] = 0
#
for i in range(plotnew.shape[0]):
	if(plotnew[i] < np.mean(plotnew)):
		plotnew[i] = 0
 
plt.figure(2)
plt.stem(plotnew)
plt.show()
#
for i in range(len(plotnew)):
	if (plotnew[i] == maxi):
		lcm = i
print(lcm)
factors = factors(lcm)
print(factors)
finalplot = []
for i in factors:
	print(i)
	finalplot.append(i*plotnew[i])
plt.figure(3)
plt.stem(factors,finalplot)
plt.show()