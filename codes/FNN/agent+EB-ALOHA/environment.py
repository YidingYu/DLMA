import numpy as np 
import random

class ENVIRONMENT(object):
	"""docstring for ENVIRONMENT"""
	def __init__(self,
				 state_size = 10,
				 window_size = 2,
				 max_backoff = 2
				 ):
		super(ENVIRONMENT, self).__init__()
		self.state_size = state_size
		self.window_size = window_size
		self.max_backoff = max_backoff

		self.count = 0 # number of collisions
		self.backoff = np.random.randint(0, self.window_size * 2**self.count)

	def reset(self):
		init_state = [0] * self.state_size
		if self.backoff == 0:
			self.aloha_action = 1
			self.backoff = np.random.randint(0, self.window_size)
		else:
			self.aloha_action = 0
		return init_state

	def update(self):
		self.count = min(self.count, self.max_backoff)
		self.backoff -= 1
		if self.backoff < 0:
			self.backoff = np.random.randint(0, self.window_size * 2**self.count)
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
				self.count = 0
		else:
			if self.aloha_action == 0:
				observation = 'S'
				reward1 = 1
			else:
				observation = 'F'
				self.count += 1

		self.update()

		return observation, reward1, reward2




