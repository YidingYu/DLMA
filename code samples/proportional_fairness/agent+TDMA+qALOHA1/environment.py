import numpy as np 

counter = 0
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
		self.n_nodes = 3
		# self.action_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # case0

		# self.action_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # case1.1
		# self.action_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # case1.2
		# self.action_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # case1.3
		
		# self.action_list = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0] # case2.1 
		# self.action_list = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0] # case2.2 
		# self.action_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1] # case2.3
	
		# self.action_list = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0] # case3.1 
		self.action_list = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0] # case3.2 
		# self.action_list = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0] # case3.3
	
		# self.action_list = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0] # case4.1 
		# self.action_list = [0, 1, 1, 0, 0, 0, 0, 0, 1, 1] # case4.2 
		# self.action_list = [0, 0, 1, 1, 1, 0, 0, 1, 0, 0] # case4.3
	
		# self.action_list = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # case5.1 
		# self.action_list = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1] # case5.2 
		# self.action_list = [0, 1, 1, 1, 0, 1, 0, 1, 0, 0] # case5.3

		# self.action_list = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0] # case6.1
		# self.action_list = [1, 1, 0, 1, 1, 0, 1, 1, 0, 0] # case6.2
		# self.action_list = [0, 1, 1, 1, 0, 1, 0, 1, 0, 1] # case6.3
	
		# self.action_list = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0] # case7.1 
		# self.action_list = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1] # case7.2
		# self.action_list = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1] # case7.3

		# self.action_list = [1, 0, 1, 1, 1, 1, 1, 1, 0, 1] # case8.1 
		# self.action_list = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] # case8.2
		# self.action_list = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1] # case8.3

		# self.action_list = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1] # case9.1
		# self.action_list = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1] # case9.2
		# self.action_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0] # case9.3

		# self.action_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # case10

	def reset(self):
		init_state = np.zeros(self.state_size, int)
		return init_state

	def step(self, action):
		global counter
		reward = 0
		agent_reward = 0
		aloha_reward = 0
		tdma_reward = 0
		observation_ = 0

		if np.random.random()<self.attempt_prob:
			aloha_action = 1
		else:
			aloha_action = 0
		tdma_action = self.action_list[counter]


		if action == 1:
			if aloha_action == 0 and tdma_action == 0:
				# print('agent success')
				reward = 1
				agent_reward = 1
				observation_ = 1 # tx, success
			else:
				# print('collision')
				observation_ = -1 # tx, no success
		else:
			if aloha_action == 1 and tdma_action == 0:
				# print('aloha success')
				reward = 1
				aloha_reward = 1
				observation_ = 2 # no tx, success
			elif aloha_action == 0 and tdma_action == 1:
				# print('tdma success')
				reward = 1
				tdma_reward = 1
				observation_ = 2 # no tx, success
			else:
				# print('idle or aloha-tdma collision')
				observation_ = -2 # no tx, no success

		counter += 1
		if counter == len(self.action_list): 
			counter = 0
		return observation_, reward, agent_reward, aloha_reward, tdma_reward




