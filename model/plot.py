import numpy as np
from matplotlib import pyplot as plt
import os

logs = []

for _ in range(100):
      ext = str(_) + '.txt'
      if 'log' + ext in os.listdir('./data/'):
            log = './data/log' + ext
            logs.append(np.loadtxt(log).reshape(-1))
            
A = np.concatenate(logs, axis=0)
A = A[np.abs(A) >= 0.]*10

plot = []
i = 0
k = 100
while i + k <= len(A):
      plot.append(np.mean(A[i:i+k]))
      i += k


plt.plot(range(len(plot)), plot)
plt.show()
