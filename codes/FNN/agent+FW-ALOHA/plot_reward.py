import numpy as np
import matplotlib.pyplot as plt


def return_throughput(rewards):
	N = int(1000)
	temp_sum = 0
	throughput = []
	for i in range(len(rewards)):
		if i < N:
			temp_sum += rewards[i]
			throughput.append(temp_sum / (i+1))
		else:
			temp_sum += rewards[i] - rewards[i-N]
			throughput.append(temp_sum / N)
	return throughput




M = 20 # state length
E = 1000 # memory size
F = 20 # target network update frequency
B = 64 # mini-batch size
gamma = 0.9 # discount factor

alpha = 1 # fairness index
max_iter = int(5e4)
idx = 1

W = 4


for idx in range(1, 2):
	reward1 = np.loadtxt(f'rewards/reward1_len{max_iter}_W{W}_M{M}_E{E}_F{F}_B{B}_gamma{gamma}_alpha{alpha}_idx{idx}.txt')
	reward2 = np.loadtxt(f'rewards/reward2_len{max_iter}_W{W}_M{M}_E{E}_F{F}_B{B}_gamma{gamma}_alpha{alpha}_idx{idx}.txt')
	throughput1 = return_throughput(reward1)
	throughput2 = return_throughput(reward2)
	print(f'agent: {np.mean(throughput1[-1000])}')
	print(f'aloha: {np.mean(throughput2[-1000])}')

	# sum_throughput = [throughput1[i]+throughput2[i] for i in range(max_iter)]
	# plt.plot(sum_throughput,label='Sum')
	
	plt.plot(throughput1, c='r', label='DLMA')
	plt.plot(throughput2, c='b', label='ALOHA')
	plt.legend()
	plt.show()


