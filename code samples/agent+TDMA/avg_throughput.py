import numpy as np
import matplotlib.pyplot as plt


def my_plot(file1, file2, t):
	max_iter = 50000
	N = 50000 

	# load reward
	agent_reward = np.loadtxt(file1)
	tdma_reward = np.loadtxt(file2)

	throughput_agent = np.zeros((1, max_iter))
	throughput_tdma = np.zeros((1, max_iter))
	agent_optimal = np.ones(max_iter)* (1-t)
	tdma_optimal = np.ones(max_iter)* t	

	agent_temp_sum = 0
	tdma_temp_sum = 0
	for i in range(0, max_iter):
		if i < N:
			agent_temp_sum += agent_reward[i]
			tdma_temp_sum  += tdma_reward[i]
			throughput_agent[0][i] = agent_temp_sum / (i+1)
			throughput_tdma[0][i]  = tdma_temp_sum / (i+1)
		else:
			agent_temp_sum += agent_reward[i] - agent_reward[i-N]
			tdma_temp_sum  += tdma_reward[i] - tdma_reward[i-N]
			throughput_agent[0][i] = agent_temp_sum / N
			throughput_tdma[0][i]  = tdma_temp_sum / N

	agent_line, = plt.plot(throughput_agent[0], color='r', lw=1.2, label='agent')
	tdma_line,  = plt.plot(throughput_tdma[0], color='b', lw=1.2, label='tdma')

	agent_optimal_line, = plt.plot(agent_optimal, color='r', lw=3, label='agent optimal')
	tdma_optimal_line,  = plt.plot(tdma_optimal, color='b', lw=3, label='tdma optimal')

	plt.xlim((0, max_iter))
	plt.ylim((-0.05, 1))
	plt.grid()
	plt.legend(handles=[agent_line, tdma_line, agent_optimal_line, tdma_optimal_line], loc='best')


for i in range(1, 2):
	plt.figure(i)
	my_plot('rewards/agent_len5e4_M20_h1_t10-3_%d.txt' % i,
		     'rewards/tdma_len5e4_M20_h1_t10-3_%d.txt' % i, t=0.3)
plt.show()


