import numpy as np
from time import time
rows = 6
columns = 7
debug=False
input_sizes = [42, 86]

def horizontal(grid, r, c):
      w = np.array([(0, 0), (0, 1), (0, 2), (0, 3)])
      return np.array([grid[r + w[i][0], c + w[i][1]] for i in range(4)])
def vertical(grid, r, c):
      w = np.array([(0, 0), (1, 0), (2, 0), (3, 0)])
      return np.array([grid[r + w[i][0], c + w[i][1]] for i in range(4)])
def diagonal(grid, r, c):
      w = np.array([(0, 0), (1, 1), (2, 2), (3, 3)])
      return np.array([grid[r + w[i][0], c + w[i][1]] for i in range(4)])
def rdiagonal(grid, r, c):
      w = np.array([[0, 0], [-1, 1], [-2, 2], [-3, 3]])
      return np.array([grid[r + w[i][0], c + w[i][1]] for i in range(4)])

def check(grid, r, c, t):
      for k in range(4):
            if c-k>=0 and c-k+3 < columns and sum(horizontal(grid, r, c-k)) == t:
                  return 1
            if r-k>=0 and r-k+3 < rows and sum(vertical(grid, r-k, c)) == t:
                  return 1
            if r-k>=0 and r-k+3 < rows and c-k>=0 and c-k+3 < columns and sum(diagonal(grid, r-k, c-k)) == t:
                  return 1
            if r-3+k>=0 and r+k < rows and c-k>=0 and c-k+3 < columns and sum(rdiagonal(grid, r+k, c-k)) == t:
                  return 1
      return 0

def normal(grid):
      return grid.reshape(6, 7, 1)

def critical(grid):

      inputs_critical = np.zeros((7,2), 'int')
      #IS FINISHED

      #horizontal
      for row in range(rows):
            for column in range(columns - 3):
                  four = horizontal(grid, row, column)
                  if sum(four) == 4:
                        inputs_critical[6][0] = 1
                  if sum(four) == -4:
                        inputs_critical[6][1] = 1
      
      #vertical
      for column in range(columns):
            for row in range(rows - 3):
                  four = vertical(grid, row, column)
                  if sum(four) == 4:
                        inputs_critical[6][0] = 1
                  if sum(four) == -4:
                        inputs_critical[6][1] = 1
      
      #diagonal
      for row in range(rows - 3):
            for column in range(columns - 3):
                  four = diagonal(grid, row, column)
                  if sum(four) == 4:
                        inputs_critical[6][0] = 1
                  if sum(four) == -4:
                        inputs_critical[6][1] = 1
                  
      for row in range(3, rows):
            for column in range(columns - 3):
                  four = rdiagonal(grid, row, column)
                  if sum(four) == 4:
                        inputs_critical[6][0] = 1
                  if sum(four) == -4:
                        inputs_critical[6][1] = 1

      
      #CRITICAL PARTS
      
      down = [rows-sum(grid[:,c]!=0)-1 for c in range(columns)]
      
      for c in range(columns): 
            for r in range(down[c], -1, -1):
                  ind = down[c]-r
                  inputs_critical[ind][0] += check(grid, r, c, 3)
                  inputs_critical[ind][1] += check(grid, r, c, -3)

      inputs_critical = np.array(inputs_critical).reshape(-1)
      
      #STRATEGIC PARTS
      inputs_strategic = np.zeros((6,6,2))


      for c in range(columns):
            for r in range(0, down[c]+1):
                  for c2 in range(c+1, columns):
                        if c2 - c >= 4:
                              break
                        
                        #horizontal
                        if grid[r][c2] == 0:
                              for c0 in range(c2-3, c+1):
                                    if c0 >= 0 and c0+3<columns and sum(horizontal(grid, r, c0)) == 2:
                                          inputs_strategic[down[c]-r][down[c2]-r][0] += 1
                                    if c0 >= 0 and c0+3<columns and sum(horizontal(grid, r, c0)) == -2:
                                          inputs_strategic[down[c]-r][down[c2]-r][1] += 1
                        #diagonal
                        r2 = r + c2 - c
                        if r2 < rows and grid[r2][c2] == 0:
                              for c0 in range(c2-3, c+1):
                                    r0 = r - (c - c0)
                                    if c0>=0 and c0+3<columns and r0>=0 and r0+3<rows and sum(diagonal(grid, r0, c0)) == 2:
                                          inputs_strategic[down[c]-r][down[c2]-r2][0] += 1
                                    if c0>=0 and c0+3<columns and r0>=0 and r0+3<rows and sum(diagonal(grid, r0, c0)) == -2:
                                          inputs_strategic[down[c]-r][down[c2]-r2][1] += 1
                                    
                        #rdiagonal
                        r2 = r - (c2 - c)
                        if r2 >= 0 and grid[r2][c2] == 0:
                              for c0 in range(c2-3, c+1):
                                    r0 = r + (c - c0)
                                    if c0>=0 and c0+3<columns and r0-3>=0 and r0<rows and sum(rdiagonal(grid, r0, c0)) == 2:
                                          inputs_strategic[down[c]-r][down[c2]-r2][0] += 1
                                    if c0>=0 and c0+3<columns and r0-3>=0 and r0<rows and sum(rdiagonal(grid, r0, c0)) == -2:
                                          inputs_strategic[down[c]-r][down[c2]-r2][1] += 1
                          
      inputs_strategic = np.array(inputs_strategic).reshape(-1)

      inputs = np.concatenate([inputs_critical, inputs_strategic], axis=0)
      inputs = np.minimum(inputs, 3)
      return inputs

def get_inputs(grid):
      grid[grid==2] = -1
      grid = grid.reshape(6, 7)
      return [normal(grid), critical(grid)]


def fit_to_model(grids):
      grids = grids.reshape(-1 ,6, 7)
      
      # assert grids.shape[0] == 1

      return [np.array([normal(grid) for grid in grids]), np.array([critical(grid) for grid in grids])]

if __name__ == "__main__":
      grid = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0,-1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1,-1, 0, 0, 0, 0],
            [-1,1,-1, 0, 0, 0, 0],
            [-1,-1,-1,1, 0, 0, 0]
      ])
      
      print(get_inputs(grid)[1].reshape(-1, 2))