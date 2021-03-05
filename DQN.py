import numpya s np
import tensorflow as tfa
from tensorflow.keras import Input, layers, Model

class DeepQNet():
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
        self.sf_num = state_feature_num
        self.action_num = action_num

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_max = epsilon
        self.epsilon = epsilon if epsilon_increment is None else 0

        self.batch_size = batch_size
        self.update_iter = update_iter
        self.update_counter = 0
        
        self.mem_size = mem_size
        self.memory = np.zeros([mem_size, state_feature_num * 2 + 2]) # [s, a, r, s_]
        self.mem_counter = 0

        self.hls = hidden_layer
        for h_num in hidden_lay:
            self.hls.append(layers.Dense(h_num, activation='relu'))

        self._build_net()

    def _build_net(self):
        en_inputs = Input(shape=(self.sf_num))
        for i in range(len(self.hls)):
            x = Dense(self.hls[i], activation='relu')(x) if i != 0 else Dense(self.hls[0], activation='relu')(en_inputs)
        self.q_eval = layers.Dense(action_num, activation='relu')(x)

        tn_inputs = Input(shape=(self.sf_num))
        for i in range(len(self.hls)):
            y = Dense(self.hls[i], activation='relu')(y) if i != 0 else Dense(self.hls[0], activation='relu')(tn_inputs)
        self.q_target = layers.Dense(action_num, activation='relu')(y)

        self.eval_net = Model(en_inputs, self.q_eval)
        self.target_net = Model(tn_inputs, self.q_target)

        self.eval_net.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
        self.target_net.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])

    def update_tn(self):
        ## update target network
        value = self.eval_net.get_weights()
        self.target_net.set_weights(value)

    def step(self, obs):
        obs = np.array(obs)[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net.predict(observation)
            action = np.argmax(actions_vale)
        else:
            action  = np.randm.randint(0, self.n_actions)
        return action

    def update_mem(self, s, a, r, s_):
        mem_unit = np.hstack(s, a, r, s_)
        mem_index = self.mem_counter % self.mem_size
        self.memory[index, :] = mem_unit
        self.mem_counter += 1

    def train(self):
        if self.copy_conter == self.copy_iter:
            self.update_tn()

        if self.mem_counter > self.mem_size:
            sample_index = np.random.choice(self.mem_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.mem_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_ = self.target_net.predict(batch_memory[:, -self.sf_num :])
        q = self.eval_net.predict(batch_memory[:, : self.sf_num])
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.sf_num].astype(int)
        reward = batch_memory[:, self.sf_num + 1]

        q_target = q.copy()
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_, axis=1)

        self.target_net.fit(batch_memory[:, : self.sf_num], q_target, epochs = 10)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.copy_conter += 1









