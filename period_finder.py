import numpy as np
import fractions as f
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy import signal
plt.close('all')

def energy(x):
	eng = np.matmul(x.T, x)
	return(eng)


x1 = 100*np.random.rand(30)
x1 = x1 - np.mean(x1)
x2 = 100*np.random.rand(12)
x2 = x2 - np.mean(x2)
x5 = 30*np.random.randn(180)
x = np.tile(x1,6) + np.tile(x2, 15) + x5
snr = 10*np.log(energy(x)/energy(x5))
print('snr', snr)
x = x[0:180]
x = x
print(np.matmul(x.T,x))
N = 160
vari = np.zeros(80)

for k in range(1,80):
	#print("\n***************************\n")
	#print("\nTesting if period is ", k)
	cumm = 0
	for i  in range(k):
		collection = []
		for j in range(N):
			if(k*j+i < N):
				collection.append(x[k*j+i])
			var = np.var(collection)
		cumm = cumm + var
	vari[k] = cumm/k
	# print("For k = ",k , "distance = ", cumm/k)
	#print("**********************************\n")
plt.figure(1)
title('Variance Plot')
xlabel('Period')
ylabel('Variance')
plt.stem(vari)
plt.show()
prev = 1
current = 2
nxt = 3
plot = []
idx = []
maxi = np.max(vari)
for i in range(1,116):
	if vari[current] < vari[prev]:
		if vari[current] < vari[nxt]:
			plot.append((maxi-vari[current])**4)
			idx.append(current)
	prev = current
	current = nxt
	nxt = nxt + 1
plt.figure(2)
title('Variance Plot')
xlabel('Period')
ylabel('Strength')
plt.stem(idx,plot)
plt.show()