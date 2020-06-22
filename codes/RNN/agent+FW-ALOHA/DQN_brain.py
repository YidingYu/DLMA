import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import random

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Add, Activation, GRU, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.activations import softmax

from collections import deque


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# set_session(tf.Session(config=config))



class DQN:
	def __init__(self, 
				features,
				n_nodes,
				n_actions,
				state_length=20,
				memory_size=500,
				replace_target_iter=200,
				batch_size=32,
				learning_rate=0.01,
				gamma=0.9,
				epsilon=1,
				epsilon_min=0.01,
				epsilon_decay=0.995,
				alpha=0
				):

		self.features = features
		self.n_nodes = n_nodes
		self.n_actions = n_actions
		self.state_length = state_length
		self.memory_size = memory_size
		self.replace_target_iter = replace_target_iter
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay   
		self.alpha = alpha

		self.memory = deque(maxlen=self.memory_size)
		
		self.learn_step_counter = 0
		self.memory_couter = 0
				
		### build mode
		self.model = self.build_RNN_model()
		self.target_model = self.build_RNN_model()

	def alpha_function(self, action_values):
		action_values_list = []
		if self.alpha == 1:
			action_values_list = [np.log(action_values[2*j]) + np.log(action_values[2*j+1])  for j in range(self.n_actions)]
		elif self.alpha == 0:
			action_values_list = [action_values[2*j] + action_values[2*j+1] for j in range(self.n_actions)]
		elif self.alpha == 100:
			action_values_list = [min(action_values[2*j], action_values[2*j+1]) for j in range(self.n_actions)]
		else:
			action_values_list = [1/(1-self.alpha) * (action_values[2*j]**(1-self.alpha) + \
								 action_values[2*j+1]**(1-self.alpha)) for j in range(self.n_actions)]
		return np.argmax(action_values_list)


	def build_RNN_model(self):
		inputs = Input(shape=(self.state_length, self.features))
		# h1 = GRU(32, activation='relu', kernel_initializer=he_normal(seed=215247), return_sequences=True)(inputs)
		# h2 = GRU(256, activation='relu', kernel_initializer=he_normal(seed=87), return_sequences=True)(inputs)
		h3 = GRU(64, activation='relu', kernel_initializer=he_normal(seed=56))(inputs)
		h4 = Dense(64, activation='relu', kernel_initializer=he_normal(seed=524))(h3)
		# h4 = Dense(256, activation='relu', kernel_initializer=he_normal(seed=217))(h4)
		# h4 = Dense(128, activation='relu', kernel_initializer=he_normal(seed=217))(h4)
		h4 = Dense(64, activation='relu', kernel_initializer=he_normal(seed=50))(h4)
		# h4 = Dense(32, activation='relu', kernel_initializer=he_normal(seed=527))(h4)
		# h4 = Dense(16, activation='relu', kernel_initializer=he_normal(seed=9005247))(h4)
		outputs = Dense(self.n_actions*self.n_nodes, kernel_initializer=he_normal(seed=89))(h4)
		model = Model(inputs=inputs, outputs=outputs)
		model.compile(loss='mse', optimizer='rmsprop')
		return model


	def choose_action(self, state):
		state = state.reshape(1, -1, self.features)
		self.epsilon *= self.epsilon_decay
		self.epsilon  = max(self.epsilon_min, self.epsilon)		
		if np.random.uniform(0, 1) < self.epsilon:
			return np.random.randint(0, self.n_actions)
		action_values = self.model.predict(state)
		return self.alpha_function(action_values[0])


	def add_experience(self, experience):
		self.memory.append(experience)


	def repalce_target_parameters(self):
		weights = self.model.get_weights()
		self.target_model.set_weights(weights)


	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.repalce_target_parameters() # iterative target model
		self.learn_step_counter += 1

		if len(self.memory) < self.memory_size and len(self.memory) > self.state_length:
			sample_index = np.random.choice(len(self.memory)-self.state_length, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_size-self.state_length, size=self.batch_size)

		batch_memory = []
		for i in range(self.batch_size):
			batch_memory.append(np.array(self.memory)[sample_index[i]:sample_index[i]+self.state_length])
		batch_memory = np.array(batch_memory)

	
		state = batch_memory[:, :, :self.features]
		action = batch_memory[:, -1, self.features].astype(int)
		reward1 = batch_memory[:, -1, self.features+1]
		reward2 = batch_memory[:, -1, self.features+2]
		next_state = batch_memory[:, :, -self.features:]

		batch_index = np.arange(self.batch_size, dtype=np.int32)

		q = self.model.predict(state)
		q_targ = self.target_model.predict(next_state)

		for i in range(self.batch_size):
			action_ = self.alpha_function(q_targ[i])
			q[i][2*action[i]]   = reward1[i] + self.gamma * q_targ[i][2*action_]
			q[i][2*action[i]+1] = reward2[i] + self.gamma * q_targ[i][2*action_+1]

		history = self.model.fit(state, q, self.batch_size, epochs=1, verbose=0)
	# def save_model(self, fn):
	# 	self.model.save(fn)