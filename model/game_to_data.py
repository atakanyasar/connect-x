import numpy as np
import os
from model import build_model
from model_input_format import get_inputs, fit_to_model, input_sizes

if not 'data' in os.listdir():
      os.mkdir('./data/')

fold = './'
for _ in range(100):
      ext = str(_) + '.txt'
      if 'log' + ext in os.listdir(fold):
            print('log'+ext, 'opened')

            X = np.loadtxt(fold+'games' + ext)
            Y = np.loadtxt(fold+'log'+ext)

            inp = fit_to_model(X)

            with open(fold+'log'+ext, 'w') as f:
                  np.savetxt(f, Y)
            with open(fold+'games'+ext, 'w') as f:
                  np.savetxt(f, inp[0].reshape(-1, 42), fmt='%i')
            with open(fold+'critical'+ext, 'w') as f:
                  np.savetxt(f, inp[1], fmt='%i')