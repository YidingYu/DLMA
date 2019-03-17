import numpy as np
import matplotlib.pyplot as plt


def my_plot(file1, file2, t=0.9):
	max_iter = 10000
	N = 1000

	# load reward
	agent1_reward = np.loadtxt(file1)
	agent2_reward = np.loadtxt(file2)

	throughput_agent1 = np.zeros((1, max_iter))
	throughput_agent2 = np.zeros((1, max_iter))


	agent1_optimal = np.ones(max_iter)* (1-t)
	agent2_optimal = np.ones(max_iter)* t	





	agent1_temp_sum = 0
	agent2_temp_sum = 0
	# agent4_temp_sum = 0
	for i in range(0, max_iter):
		if i < N:
			agent1_temp_sum += agent1_reward[i]
			agent2_temp_sum += agent2_reward[i]
			# agent4_temp_sum += agent4_reward[i]
			throughput_agent1[0][i] = agent1_temp_sum / (i+1)
			throughput_agent2[0][i] = agent2_temp_sum / (i+1)
		else:
			agent1_temp_sum += agent1_reward[i] - agent1_reward[i-N]
			agent2_temp_sum += agent2_reward[i] - agent2_reward[i-N]
			throughput_agent1[0][i] = agent1_temp_sum / N
			throughput_agent2[0][i] = agent2_temp_sum / N

	plt.xlim((0, max_iter))
	plt.ylim((-0.05, 1))

	agent1_line, = plt.plot(throughput_agent1[0], color='r', lw=1.2, label='agent')
	agent2_line, = plt.plot(throughput_agent2[0], color='b', lw=1.2, label='aloha')

	agent3_line, = plt.plot(agent1_optimal, color='r', lw=3, label='agent optimal')
	agent4_line, = plt.plot(agent2_optimal, color='b', lw=3, label='aloha optimal')
	plt.grid()
	# # plt.legend(handles=[agent1_line, agent2_line, agent3_line, agent4_line], loc='best')
	# print('---------------')
	# print('agent', np.mean(throughput_agent1[0][-100:]))
	# print('tdma', np.mean(throughput_agent2[0][-100:]))

for i in range(1, 5):
	plt.figure(i)
	my_plot('rewards/agent_len1e5_M20_h6_t10-8_%d.txt' % i,
		     'rewards/tdma_len1e5_M20_h6_t10-8_%d.txt' % i, t=0.8)
plt.show()

