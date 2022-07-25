import numpy as np
from time import time
from tensorflow import keras

from model import build_model
from model_input_format import fit_to_model, get_inputs

debug=False

def reinforced_agent(observation, configuration):
      model = build_model()
      model.load_weights('model.h5')

      begin = time()

      rows = configuration.rows
      columns = configuration.columns
      
      def drop(grid, col, player):
            new_grid = grid.copy()
            for r in range(rows-1, -1, -1):
                  if grid[r][col] == 0:
                        new_grid[r][col] = player
                        return new_grid
            assert 0

      
      def possible_moves(grid):
            return np.array([grid[0][c] == 0 for c in range(columns)])


      grid = np.asarray(observation.board).reshape(configuration.rows, configuration.columns)
      grid[grid[:,:] == 2] = -1

      us = observation.mark
      if debug:
            print(np.array(model(fit_to_model(grid.copy()))))
      if us == 1:
            grid[grid==-1] = 2
            grid[grid==1] = -1
            grid[grid==2] = 1
      
      ret = 1.*possible_moves(grid)

      eps = 0.05
      
      if debug:
            eps=0

      if np.random.rand() < eps:
            for c in range(7):
                  if ret[c] == 1:
                        ret[c] = np.random.rand()
                  else:
                        ret[c] = -100
      else:
            for c in range(7):
                  if ret[c] == 1:
                        ret[c] = -model(fit_to_model(drop(grid, c, -1)))
                        if debug:
                              print(get_inputs(drop(grid, c, -1))[1][:14])
                              print(ret[c])
                  else:
                        ret[c] = -100
      if debug:
            print(time()-begin, "second")
      
      return int(np.argmax(ret))