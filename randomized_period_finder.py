import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal
import random
plt.close('all')
# x1 = 100*signal.triang(7)
# x2 = 100*np.random.rand(13)
x3 = 100*signal.cosine(7)
x4 = 100*signal.triang(19)
x =  50*np.random.rand(7*19*8) + np.tile(x3, 19*8) + np.tile(x4, 7*8)
x = x[0:297]
print(np.matmul(x.T,x))
N = x.shape[0]
vari = np.zeros(N)

for p in range(10):
	va = 0
	for k in range(2,int(N/2)):
		#print("\n***************************\n")
		#print("\nTesting if period is ", k)
		cumm = 0
		if(k > 50):
			randz = random.sample(range(0, k), 50)
		else:
		 	randz = range(k)
		for i  in randz:
			collection = []
			if(int(N/k) > 10):
				equi_class = random.sample(range(0, int(N/k)), 10)
			else: 
				equi_class = range(int(N/k)) 
			for j in equi_class :
				collection.append(x[k*j+i])
				var = np.var(collection)
				############################################################################
				# This code is very fast and gives the same result as always :)            #
				############################################################################
				#Find the appropriate factor so that hidden periods are also weighted well #
				############################################################################
				# I used two step penelization one at the var level and the other at the   # 
				# cumm level                                                               #
				############################################################################
			cumm = cumm + (N/k)*var #/(len(equi_class)**3) #### This is the first level you need to change powers #######
		vari[k] = cumm #/((len(randz))**3) #### This is the second level .... here also ##########
		#print("For k = ",k , "distance = ", cumm)
		#print("**********************************\n")
	for idx,val in enumerate(vari):
		if idx == 0:
			pass
		else:
			if ((vari[idx] is 0) and (vari[int(0.5*idx)] is not 0)) or ((vari[idx] is not 0) and (vari[int(0.5*idx)] is 0)):
				vari[idx] = 0
				vari[int(0.5*idx)] = 0
	varii = vari
	va = va + varii

vari = va
plt.figure(1)
plt.stem(vari)
plt.show()
prev = 1
current = 2
nxt = 3
plot = []
idx = []
max_var = np.max(vari)
for i in range(1,int(N/2)-2):
	if vari[current] < vari[prev]:
		if vari[current] < vari[nxt]:
			plot.append(max_var - vari[current])
			idx.append(current)
	prev = current
	current = nxt
	nxt = nxt + 1
plt.figure(2)
plt.stem(idx,plot)
plt.show()
