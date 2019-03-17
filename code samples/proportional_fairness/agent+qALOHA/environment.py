import numpy as np 

class ENVIRONMENT(object):
	"""docstring for ENVIRONMENT"""
	def __init__(self,
				 state_size = 10,
				 attempt_prob = 1,
				 ):
		super(ENVIRONMENT, self).__init__()
		self.state_size = state_size
		self.attempt_prob = attempt_prob # aloha node attempt probability
		self.action_space = ['w', 't'] # w: wait t: transmit
		self.n_actions = len(self.action_space)
		self.n_nodes = 2
		
	def reset(self):
		init_state = np.zeros(self.state_size, int)
		return init_state

	def step(self, action):
		reward = 0
		agent_reward = 0
		aloha_reward = 0
		observation_ = 0

		if np.random.random()<self.attempt_prob:
			aloha_action = 1
		else:
			aloha_action = 0

		if action == 1:
			if aloha_action == 1:
				# print('collision')
				observation_ = -1 # tx, no success
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
			else:
				# print('idle')
				observation_ = -2 # no tx, no success

		return observation_, reward, agent_reward, aloha_reward




