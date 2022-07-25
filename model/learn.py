import numpy as np
import random
import os
from time import time

from agent import my_agent
from reinforced_agent import reinforced_agent
from model_input_format import get_inputs, input_sizes
from model import build_model

rows = 6
columns = 7
iterations = 100
wait = 2

class Observation:
      def __init__(self, board, mark):
            self.board = board
            self.mark = mark
class Configuration:
      def __init__(self, row, column):
            self.rows = row
            self.columns = column

def reward(grid, player):
      #horizontal
      for r in range(rows):
            for c in range(columns-3):
                  if np.sum(grid[r,c:c+4] == player) == 4:
                        return 1
      #vertical
      for r in range(rows-3):
            for c in range(columns):
                  if np.sum(grid[r:r+4,c] == player) == 4:
                        return 1
      #diagonal
      for r in range(rows-3):
            for c in range(columns-3):
                  sum = 0
                  for i in range(4):
                        sum += grid[r+i][c+i] == player
                  if sum == 4:
                        return 1
      
      for r in range(rows-3):
            for c in range(columns-3):
                  sum = 0
                  for i in range(4):
                        sum += grid[r+i][c+3-i] == player
                  if sum == 4:
                        return 1
      return 0

def finished(grid):
      return reward(grid, 1) == 1 or reward(grid, -1) == 1 or np.sum(grid.reshape(-1) == 0) == 0

def drop(grid, col, player):
      for r in range(5, -1, -1):
            if grid[r][col] == 0:
                  board[r][col] = player
                  break

def possible_moves(grid):
      return [grid[0][c] == 0 for c in range(columns)]

def reverse_board(board, player):
      grid = board.copy()
      if player != 1:
            grid[grid==-1] = 2
            grid[grid==1] = -1
            grid[grid==2] = 1
      return grid

num = 0

model = build_model()
model.load_weights('model.h5')

if not 'data' in os.listdir():
      os.mkdir('./data/')
for _ in range(100):
      ext = str(_) + '.txt'
      if not 'log' + ext in os.listdir('./data/'):
            print('log'+ext)
            log = './data/log' + ext
            data = ['./data/games'+ext, './data/critical'+ext]
            break


working_time = 0

reinforcement = [[], [], []]

for iteration in range(1, iterations+1):
      print("game", iteration, end=': ')
      begin = time()
      board = np.zeros((rows, columns), dtype='int')
      
      if iteration % 2 == 1:
            agents = [my_agent, reinforced_agent]
            we = -1
      else:
            agents = [reinforced_agent, my_agent]
            we = 1
            
      done = 0
      player = 1
      remaining_steps = 42

      records = [[], []]

      while done == 0 and remaining_steps>0:
            # t = time()
            move = agents[player-1](Observation(board.copy(), player), Configuration(rows, columns))
            
            # print(time()-t)
            if possible_moves(board)[move]:
                  drop(board, move, player)
                  done = reward(board, player)
                        
                  if done != 0:
                        if player == 2:
                              done = -1
            else:
                  if player == 1:
                        done = -1
                  else:
                        done = 1
            player = 3 - player
            remaining_steps -= 1

            I = get_inputs(reverse_board(board, player))
            for i in range(2):
                  records[i].append(I[i])
                  reinforcement[i].append(I[i])
      
      player = -1
      score = done * (0.9+0.1*remaining_steps/42)
      
      Y = []
      for i in range(41,remaining_steps-1,-1):
            print(score*(0.9**(i-remaining_steps))*player, file=open(log, 'a'))
            reinforcement[2].append([score*(0.9**(i-remaining_steps))*player])
            Y.append([score*(0.9**(i-remaining_steps))*player])
            player = -player

      Y = np.array(Y).reshape(42-remaining_steps, 1)

      print("")
      for w in range(2):
            h = model.train_on_batch([np.array(records[i]) for i in range(2)], Y.reshape(-1, 1), reset_metrics=True)
            
            print('loss', h)
      
      for i in range(2):
            with open(data[i], 'a') as f:
                  np.savetxt(f, np.array(records[i]).reshape(-1, input_sizes[i]), fmt='%i')

      print('winner:',done*we, end=' ')
      print('remainig steps:', remaining_steps, end=' ')
      print(time()-begin, 'second,', end=' ')
      working_time += time()-begin
      print('expected remaining time:', (iterations-iteration)*(working_time/iteration)/60, 'minutes')

      if iteration % wait == 0:
            model.save_weights('model.h5')

print('looking records...')
model.fit([np.array(reinforcement[i]) for i in range(2)], np.array(reinforcement[2]), batch_size=256, epochs=10, verbose=2)
model.save_weights('model.h5')