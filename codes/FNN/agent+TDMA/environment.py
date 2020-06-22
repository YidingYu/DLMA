import numpy as np 



class ENVIRONMENT(object):
	def __init__(self,
				 state_size=10,
				 ):
		self.state_size = state_size   
		self.count = 0

		# self.tdma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # case0

		# self.tdma = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # case1.1
		# self.tdma = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # case1.2
		# self.tdma = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # case1.3
		
		# self.tdma = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0] # case2.1 
		# self.tdma = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0] # case2.2
		# self.tdma = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1] # case2.3
	
		# self.tdma = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0] # case3.1 
		self.tdma = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0] # case3.2 
		# self.tdma = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0] # case3.3
	
		# self.tdma = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0] # case4.1 
		# self.tdma = [0, 1, 1, 0, 0, 0, 0, 0, 1, 1] # case4.2 
		# self.tdma = [0, 0, 1, 1, 1, 0, 0, 1, 0, 0] # case4.3
	
		# self.tdma = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # case5.1 
		# self.tdma = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1] # case5.2 
		# self.tdma = [0, 1, 1, 1, 0, 1, 0, 1, 0, 0] # case5.3

		# self.tdma = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0] # case6.1
		# self.tdma = [1, 1, 0, 1, 1, 0, 1, 1, 0, 0] # case6.2
		# self.tdma = [0, 1, 1, 1, 0, 1, 0, 1, 0, 1] # case6.3
	
		# self.tdma = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0] # case7.1 
		# self.tdma = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1] # case7.2
		# self.tdma = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1] # case7.3

		# self.tdma = [1, 0, 1, 1, 1, 1, 1, 1, 0, 1] # case8.1 
		# self.tdma = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] # case8.2
		# self.tdma = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1] # case8.3

		# self.tdma = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1] # case9.1
		# self.tdma = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1] # case9.2
		# self.tdma = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0] # case9.3

		# self.tdma = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # case10
		
		# self.tdma = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0] # case2.1+6.1
		# self.tdma = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0] # case2.2+6.2
		# self.tdma = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1] # case2.3+6.3
		# self.tdma = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0] # case3.1+7.1
		# self.tdma = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1] # case3.2+7.2
		# self.tdma = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1] # case3.3+7.3
		
		# self.tdma = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0] # case5.1+7.1
		# self.tdma = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1] # case5.2+7.2
		# self.tdma = [0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1] # case5.3+7.3
		
		# self.tdma = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0] # case3.1+5.1+7.1
		# self.tdma = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1] # case3.1+5.2+7.2
		# self.tdma = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1] # case3.3+5.3+7.3


	def reset(self):
		self.tdma_action = self.tdma[self.count]
		init_state = [0] * self.state_size
		return init_state

	def update(self):
		self.count += 1
		if self.count == len(self.tdma):
			self.count = 0
		self.tdma_action = self.tdma[self.count]


	def step(self, action):

		reward1, reward2 = 0, 0
		if action == 0:
			if self.tdma_action == 0:
				observation = 'I'
			else:
				observation = 'B'
				reward2 = 1
		else:
			if self.tdma_action == 0:
				observation = 'S'
				reward1 = 1
			else:
				observation = 'F'

		self.update()
		return observation, reward1, reward2 