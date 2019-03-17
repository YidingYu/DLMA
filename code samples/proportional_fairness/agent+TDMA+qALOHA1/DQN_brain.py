import numpy as np
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Add, Activation, BatchNormalization
from keras.optimizers import RMSprop
from keras.initializers import glorot_normal


class DQN:
	def __init__(self, 
				state_size,
				n_actions,
				n_nodes,
				memory_size=500,
				replace_target_iter=200,
				batch_size=32,
				learning_rate=0.01,
				gamma=0.9,
				epsilon=1,
				epsilon_min=0.01,
				epsilon_decay=0.995,
				):
		# hyper-parameters
		self.state_size = state_size
		self.n_actions = n_actions
		self.n_nodes = n_nodes
		self.memory_size = memory_size
		self.replace_target_iter = replace_target_iter
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay       
		self.memory = np.zeros((self.memory_size, self.state_size * 2 + (self.n_nodes+1))) # memory_size * len(s, a, r, s_)
		# temporary parameters
		self.learn_step_counter = 0
		self.memory_couter = 0
				
		# # # # # # # build mode
		self.model        = self.build_ResNet_model() # model: evaluate Q value
		self.target_model = self.build_ResNet_model() # target_mode: target network

	def build_ResNet_model(self):
		inputs = Input(shape=(self.state_size, ))
		h1 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=247))(inputs) #h1
		h2 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=2407))(h1) #h2

		h3 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=2403))(h2) #h3
		h4 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=24457))(h3) #h4
		add1 = Add()([h4, h2])
		
		h5 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=24657))(add1) #h5
		h6 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=27567))(h5) #h6
		add2 = Add()([h6, add1])

		# h7 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=24657))(add2) #h5
		# h8 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=27567))(h7) #h6
		# add3 = Add()([h7, add2])

		outputs =  Dense(self.n_actions*self.n_nodes, kernel_initializer=glorot_normal(seed=242147))(add2)
		model = Model(inputs=inputs, outputs=outputs)
		model.compile(loss="mse", optimizer=RMSprop(lr=self.learning_rate))
		return model

	def choose_action(self, state):
		state = state[np.newaxis, :]
		self.epsilon *= self.epsilon_decay
		self.epsilon  = max(self.epsilon_min, self.epsilon)
		if np.random.random() < self.epsilon:
			return np.random.randint(0, self.n_actions)
		action_values = self.model.predict(state)
		action_values1 = np.log(action_values[0][0]) + np.log(action_values[0][1]) + np.log(action_values[0][2])
		action_values2 = np.log(action_values[0][3]) + np.log(action_values[0][4]) + np.log(action_values[0][5])
		action_values  = [action_values1, action_values2]
		return np.argmax(action_values)

	def store_transition(self, s, a, r1, r2, r3, s_): # s_: next_state
		if not hasattr(self, 'memory_couter'):
			self.memory_couter = 0
		transition = np.concatenate((s, [a, r1, r2, r3], s_))
		index = self.memory_couter % self.memory_size
		self.memory[index, :] = transition
		self.memory_couter   += 1


	def repalce_target_parameters(self):
		weights = self.model.get_weights()
		self.target_model.set_weights(weights)


	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.repalce_target_parameters() # iterative target model
		self.learn_step_counter += 1

		if self.memory_couter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_couter, size=self.batch_size)        
		batch_memory = self.memory[sample_index, :]

		state      = batch_memory[:, :self.state_size]
		action     = batch_memory[:, self.state_size].astype(int) # float -> int
		reward1    = batch_memory[:, self.state_size+1]
		reward2    = batch_memory[:, self.state_size+2]
		reward3    = batch_memory[:, self.state_size+3]
		next_state = batch_memory[:, -self.state_size:]


		q_eval = self.model.predict(state) # state
		
		q_next = self.target_model.predict(next_state) # next state

		batch_index = np.arange(self.batch_size, dtype=np.int32)
		q_target = q_eval.copy()
		for i in range(len(action)):
			action_values1_ = np.log(q_next[0][0]) + np.log(q_next[0][1]) + np.log(q_next[0][2])
			action_values2_ = np.log(q_next[0][3]) + np.log(q_next[0][4]) + np.log(q_next[0][5])
			action_values_ = [action_values1_, action_values2_]
			action_ = np.argmax(action_values_)    
			if action[i]==0:
				if action_==0:
					q_target[i][0] = reward1[i] + self.gamma * q_next[i][0]
					q_target[i][1] = reward2[i] + self.gamma * q_next[i][1]
					q_target[i][2] = reward3[i] + self.gamma * q_next[i][2]
				else:
					q_target[i][0] = reward1[i] + self.gamma * q_next[i][3]
					q_target[i][1] = reward2[i] + self.gamma * q_next[i][4]
					q_target[i][2] = reward3[i] + self.gamma * q_next[i][5]                    
			else:
				if action_==0:
					q_target[i][3] = reward1[i] + self.gamma * q_next[i][0]
					q_target[i][4] = reward2[i] + self.gamma * q_next[i][1]
					q_target[i][5] = reward3[i] + self.gamma * q_next[i][2]
				else:
					q_target[i][3] = reward1[i] + self.gamma * q_next[i][3]
					q_target[i][4] = reward2[i] + self.gamma * q_next[i][4]
					q_target[i][5] = reward3[i] + self.gamma * q_next[i][5]   
		self.model.fit(state, q_target, self.batch_size, epochs=1, verbose=0)

	def save_model(self, fn):
		self.model.save(fn)