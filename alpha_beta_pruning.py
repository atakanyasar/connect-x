import numpy as np

def mean(lis):
      if len(lis) == 0:
            return 0
      else:
            return sum(lis) / len(lis)

class GameTree:
      class Node:
            def __init__(self, depth=0, player=1, value=0., actions=[], parent=None, name=None):
                  self.depth = depth
                  self.player = player
                  self.value = value
                  self.name = name
                  self.actions = actions
                  self.parent = parent
                  self.edges = []
            def finished(self):
                  if self.player == 1:
                        return (self.value>=0.75) or (self.value<=-0.01)
                  else:
                        return (self.value>=0.01) or (self.value<=-0.75)
            def debug(self):
                  if self.value < -10 or self.value > 10:
                        return 
                  if not((len(self.actions)<=0 or self.actions[0] == 0) and (len(self.actions)<=1 or self.actions[1] == 5) and (len(self.actions)<=2 or self.actions[2] == 0) and (len(self.actions)<=3 or self.actions[3] == 3) and (len(self.actions)<=4 or self.actions[4] == 0)):
                        return
                  print(self.name, self.value, self.player, self.actions, end=' ')
                  if self.name != 0:
                        print(self.parent.name)
                  else:
                        print("")

      def __init__(self, children, initial_state, initializer=3, max_depth=10, value_function=None, action_function=None):
            self.nodes = {}
            self.total_nodes = 0
            self.children = children
            self.max_depth = max_depth
            self.value_function = value_function
            self.action_function = action_function
            self.initial_state = initial_state
            self.root = self.new_node(actions=[])

            id = 0
            for i in range(initializer+1):
                  id += pow(self.children, i)
            
            for i in range(id):
                  if i == len(self.nodes):
                        break
                  if self.nodes[i].finished():
                        continue
                  self.access(self.nodes[i])
            

      def new_node(self, depth=0, value=0., actions=[], parent=None):
            name = self.total_nodes
            self.total_nodes += 1
            self.nodes[name] = self.Node(depth=depth, player=1-depth%2*2, value=value, actions=actions, parent=parent, name=name)
            return self.nodes[name]

      def expand(self, node):
            if node.finished() == False and len(node.edges) == 0:
                  values = self.value_function(self.initial_state, node.actions)
                  node.edges = [self.new_node(node.depth+1, value=values[i], actions=self.action_function(node.actions, i), parent=node) for i in range(self.children)]

      def update(self, node):
            old_val = node.value
            if len(node.edges) != 0:
                  if node.player == 1:
                        node.value = max([n.value for n in node.edges])
                  else:
                        node.value = min([n.value for n in node.edges]) * 0.9 + mean([n.value for n in node.edges if n.value < 2 and n.value > -2])/7 * 0.1
            # if(old_val == node.value): 
                  # return
            if node.parent != None:
                  self.update(node.parent)

      def access(self, node):
            if not node.finished():
                  self.expand(node)
                  self.update(node)
            return node

      def best_action(self, node):
            self.access(node)
            if node.player == 1: 
                  return np.argmax([n.value for n in node.edges])
            else:
                  return np.argmin([n.value for n in node.edges])

      def play(self, games):
            for _ in range(games):
                  node = self.root
                  for __ in range(self.max_depth-1):
                        if node.finished():
                              break
                        if np.random.randn() < 2/self.max_depth:
                              a = np.random.randint(0, 7)
                        else:
                              a = self.best_action(node)
                        node = node.edges[a]
                        self.access(node)

            return self.best_action(self.root)

      def __getitem__(self, key):
            return self.access(self.nodes[key])
      
      def debug(self):
            for name in self.nodes:
                  self.nodes[name].debug()
      def dfs(self, node):
            node.debug()
            for n in node.edges:
                  self.dfs(n)

def smart_agent(observation, configuration):
      import numpy as np
      import time
      begin = time.time()
      
      rows = configuration.rows
      columns = configuration.columns


      
      def micro_reward(grid, r, c, player):
            ptr1 = c-1
            ptr2 = c+1
            while ptr1 >= 0 and grid[r][ptr1] == player:
                  ptr1 -= 1
            while ptr2 < columns and grid[r][ptr2] == player:
                  ptr2 += 1
            score1 = ptr2 - ptr1 - 1
            
            ptr1 = r-1
            ptr2 = r+1
            while ptr1 >= 0 and grid[ptr1][c] == player:
                  ptr1 -= 1
            while ptr2 < rows and grid[ptr2][c] == player:
                  ptr2 += 1
            score2 = ptr2 - ptr1 - 1
            
            ptr1 = -1
            ptr2 = +1
            while r+ptr1>=0 and c+ptr1>=0 and grid[r+ptr1][c+ptr1] == player:
                  ptr1 -= 1
            while r+ptr2<rows and c+ptr2<columns and grid[r+ptr2][c+ptr2] == player:
                  ptr2 += 1
            score3 = ptr2 - ptr1 - 1
            
            ptr1 = -1
            ptr2 = +1
            while r+ptr1>=0 and c-ptr1<columns and grid[r+ptr1][c-ptr1] == player:
                  ptr1 -= 1
            while r+ptr2<rows and c-ptr2>=0 and grid[r+ptr2][c-ptr2] == player:
                  ptr2 += 1
            score4 = ptr2 - ptr1 - 1
            
            if max(score1, score2, score3, score4) >= 4:
                  return 1.*player
            else:
                  return 0.0001 * (score1 + score2 + score3 + score4) * player

      
      def reward(grid, actions):
            trace = []
            score = 0
            player = 1

            for a in actions:
                  r, c = drop(grid, a, player)
                  if r == -1:
                        score = -100.*player
                        break
                  trace.append([r, c])
                  grid[r][c] = player
                  player_reward = micro_reward(grid, r, c, player)
                  
                  if player_reward == player:
                        score = player
                        break
                  else:
                        score += player_reward

                  player = -player

            for r, c in trace:
                  grid[r][c] = 0
            return score

      def drop(grid, col, player):
            for r in range(5, -1, -1):
                  if grid[r][col] == 0:
                        return r, col
            return -1, -1

      def reverse_board(grid):
            grid[grid[:,:]==1] = 2
            grid[grid[:,:]==-1] = 1
            grid[grid[:,:]==2] = -1

      def possible_moves(grid):
            return [grid[0][c] == 0 for c in range(columns)]


      grid = np.asarray(observation.board).reshape(configuration.rows, configuration.columns)
      grid[grid[:,:] == 2] = -1
      us = observation.mark
      
      if us == 2:
            reverse_board(grid)
            us = 1
      
      def action_function(actions, a):
            new_actions = actions.copy()
            new_actions.append(a)
            return new_actions

      def value_function(grid, actions):
            poss = 1.*np.array(possible_moves(grid))
            
            for i in range(7):
                  poss[i] = reward(grid, action_function(actions, i)) + np.random.randn()*0.00001
            return poss
      
      init = 4
      if np.random.rand() < 0.15:
            init = 5

      T = GameTree(7, grid, initializer=init, max_depth=7, value_function=value_function, action_function=action_function)
      move = T.play(0)
      end = time.time()
      
      # T.dfs(T.root)
      # T.debug()
      
      ret = 1.0*np.array(possible_moves(grid))
      ret[move] += 0.1

      # print(end-begin, "second")
      return int(np.argmax(ret))