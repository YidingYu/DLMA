import numpy as np 
import random

class ENVIRONMENT(object):
	"""docstring for ENVIRONMENT"""
	def __init__(self,
				 state_size = 10,
				 window_size = 1,
				 ):
		super(ENVIRONMENT, self).__init__()
		self.state_size = state_size
		self.window_size = window_size
		self.action_space = ['w', 't'] # w: wait t: transmit
		self.n_actions = len(self.action_space)
		self.n_nodes = 2
		self.max_retry = 2
		
	def reset(self):
		init_state = np.zeros(self.state_size, int)
		ENVIRONMENT.aloha_list = np.zeros(1000000, int)
		index = random.randint(0, self.window_size-1)
		ENVIRONMENT.aloha_list[index] = 1
		return init_state

	def step(self, action, i):
		reward = 0
		agent_reward = 0
		aloha_reward = 0
		observation_ = 0

		if ENVIRONMENT.aloha_list[i] == 0:
			aloha_action = 0
		else:
			aloha_action = 1

		if action == 1:
			if aloha_action == 1:
				# print('collision')
				observation_ = -1 # tx, no success
				if ENVIRONMENT.aloha_list[i] < self.max_retry:
					index =i+1+random.randint(0, self.window_size*(2**ENVIRONMENT.aloha_list[i]) - 1)
				else:
					index = i+1+random.randint(0, self.window_size*(2**self.max_retry) - 1)
				ENVIRONMENT.aloha_list[index] = ENVIRONMENT.aloha_list[i] + 1
			else:
				# print('agent success')
				reward = 1
				agent_reward = 1
				observation_ = 1 # tx, success
		else:
			if aloha_action == 1:
				# print('aloha success')
				reward = 1
				aloha_reward = 1
				observation_ = 2 # no tx, success
				index = i+1+random.randint(0, self.window_size-1)
				ENVIRONMENT.aloha_list[index] = 1
			else:
				# print('idle')
				observation_ = -2 # no tx, no success

		return observation_, reward, agent_reward, aloha_reward




