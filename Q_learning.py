import numpy as np

class QLearning():
    def __init__(self, state_num, action_num, epsilon=0.9, alpha=0.1, gamma=0.9):
        self.state_num = state_num
        self.action_num = action_num
        self.q_table = np.zeros((state_num, action_num), dtype=np.float32)
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
    def step(self, s):
        self.s = s
        candidate_a = self.q_table[s, :]
        if (np.random.uniform() > self.epsilon) or ((candidate_a == 0).all()):
            action = np.random.choice(ACTIONS)
        else:
            action = candidate_a.idxmax()
        self.a = a
        return action
    
    def train(self, s_, r):
        if s_ == -1:
            q_target = r
        else: 
            q_target = r + self.gamma * self.q_table[s_, :].max()
        
        self.q_table[self.s, self.a] += self.alpha * (q_target - self.q_table[self.s, self.a])
    
    def save_model(self, path):
        np.save(path, self.q_table)
        
    def load_model(self, path):
       self.q_table = np.load(path)