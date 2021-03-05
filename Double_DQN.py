import numpya s np
import tensorflow as tfa
from tensorflow.keras import Input, layers, Model
from DQN import DeepQNet

class DoubleDQN(DQN):
    def __init__(
            self,
            state_feature_num,
            action_num,
            alpha=0.01,
            gamma=0.9,
            epsilon=0.9,
            epsilon_increment=None,
            batch_size=32,
            update_iter=100,
            mem_size=500,
            hidden_layer=[]
    ):
        superg(DoubleDQN, self).__init__(
            state_feature_num,
            action_num,
            alpha=0.01,
            gamma=0.9,
            epsilon=0.9,
            epsilon_increment=None,
            batch_size=32,
            update_iter=100,
            mem_size=500,
            hidden_layer=[]
        )

    def train():
        if self.copy_conter == self.copy_iter:
            self.update_tn()

        if self.mem_counter > self.mem_size:
            sample_index = np.random.choice(self.mem_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.mem_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        tn_q_ = self.target_net.predict(batch_memory[:, -self.sf_num :])
        en_q_ = self.eval_net.predict(batch_memory[:, -self.sf_num :])
        q = self.eval_net.predict(batch_memory[:, : self.sf_num])
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.sf_num].astype(int)
        reward = batch_memory[:, self.sf_num + 1]

        q_target = q.copy()
        action_i = np.argmax(en_q_, axis=1)
        q_target[batch_index, eval_act_index] = reward + self.gamma * tn_q_[batch_index, action_i]

        self.target_net.fit(batch_memory[:, : self.sf_num], q_target, epochs = 10)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.copy_conter += 1
