import numpy as np 
import random

class ENVIRONMENT(object):
	"""docstring for ENVIRONMENT"""
	def __init__(self,
				 features = 10,
				 window_size = 2,
				 ):
		super(ENVIRONMENT, self).__init__()
		self.features = features
		self.window_size = window_size
		self.backoff = np.random.randint(0, self.window_size)

	def reset(self):
		init_state = [0] * self.features
		if self.backoff == 0:
			self.aloha_action = 1
			self.backoff = np.random.randint(0, self.window_size)
		else:
			self.aloha_action = 0
		return init_state

	def update(self):
		self.backoff -= 1
		if self.backoff < 0:
			self.backoff = np.random.randint(0, self.window_size)
		if self.backoff == 0:
			self.aloha_action = 1
		else:
			self.aloha_action = 0


	def step(self, action):

		reward1, reward2 = 0, 0
		if action == 0:
			if self.aloha_action == 0:
				observation = 'I'
			else:
				observation = 'B'
				reward2 = 1
		else:
			if self.aloha_action == 0:
				observation = 'S'
				reward1 = 1
			else:
				observation = 'F'

		self.update()

		return observation, reward1, reward2




