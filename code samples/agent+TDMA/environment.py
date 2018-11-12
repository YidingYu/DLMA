import numpy as np 

counter = 0

class ENVIRONMENT(object):
	"""docstring for ENVIRONMENT"""
	counter = 0
	def __init__(self,
				 state_size=10,
				 ):
		self.state_size = state_size   
		self.action_space = ['w', 't'] # wait transmit
		self.n_actions = len(self.action_space)
		self.n_nodes = 2

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

	def step(self,action):
		global counter
		tdma_reward = 0
		agent_reward = 0 
		reward = 0
		observation_ = 0
		tdma_action = self.action_list[counter]

		if action == 1:
			if tdma_action == 1:
				# print('collision')
				observation_ = -1 # tx, no success
			else:
				# print('agent success')
				reward = 1
				agent_reward = 1
				observation_ = 1 # tx, success
		else:
			if tdma_action == 1:
				# print('tdma success')
				reward = 1
				tdma_reward = 1
				observation_ = 1 # no tx, success
			else:
				# print('idle')
				observation_ = -2 # no tx, no success

		counter += 1
		if counter == len(self.action_list): 
			counter = 0
		return observation_, reward, agent_reward, tdma_reward 