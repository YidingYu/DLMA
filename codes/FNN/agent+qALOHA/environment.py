import numpy as np 



class ENVIRONMENT(object):
	def __init__(self,
				 state_size=10,
				 tx_prob=0.2
				 ):
		self.state_size = state_size   
		self.tx_prob = tx_prob

	def reset(self):
		self.aloha_action = 0
		init_state = [0] * self.state_size
		return init_state

	def update(self):
		if np.random.uniform(0, 1) < self.tx_prob:
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