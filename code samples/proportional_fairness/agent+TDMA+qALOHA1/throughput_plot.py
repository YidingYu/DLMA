import numpy as np
import matplotlib.pyplot as plt





### calculate throughput
def cal_throughput(max_iter, N, reward):
	temp_sum = 0
	throughput = np.zeros(max_iter)
	for i in range(max_iter):
		if i < N:
			temp_sum     += reward[i] 
			throughput[i] = temp_sum / (i+1)
		else:
			temp_sum  += reward[i] - reward[i-N]
			throughput[i] = temp_sum / N
	return throughput


agent_rewards = {}
aloha_rewards = {}
# tdma_rewards = {}
agent_throughputs = {}
aloha_throughputs = {}
# tdma_throughputs = {}
max_iter = 100000
N = 10000
x = np.linspace(0, max_iter, max_iter)
q = 0.2
for i in range(0, 4):
	agent_rewards[i]     = np.loadtxt('rewards/agent_len1e5_M20_h30_q0.2_%d.txt' % (i+1))
	aloha_rewards[i]     = np.loadtxt('rewards/aloha_len1e5_M20_h30_q0.2_%d.txt' % (i+1))
	# tdma_rewards[i]      = np.loadtxt('rewards/tdma_len1e6_M4_q0.4_t4-1_%d.txt' % (i+1))
	agent_throughputs[i] = cal_throughput(max_iter, N, agent_rewards[i])
	aloha_throughputs[i] = cal_throughput(max_iter, N, aloha_rewards[i])
	# tdma_throughputs[i]  = cal_throughput(max_iter, N, tdma_rewards[i])

agent_optimal = np.ones(max_iter)*((1-q)/2)
aloha_optimal = np.ones(max_iter)*(q/2)
# tdma_optimal  = np.ones(max_iter)*(0.25*(1-q)**2)


my_agent_throughputs = np.array([agent_throughputs[0], agent_throughputs[1], agent_throughputs[2], agent_throughputs[3]])
my_aloha_throughputs = np.array([aloha_throughputs[0], aloha_throughputs[1], aloha_throughputs[2], aloha_throughputs[3]])
# my_tdma_throughputs  = np.array([tdma_throughputs[0], tdma_throughputs[1], tdma_throughputs[2], tdma_throughputs[3]])




agent_mean = np.mean(my_agent_throughputs, axis=0)
aloha_mean = np.mean(my_aloha_throughputs, axis=0)
# tdma_mean  = np.mean(my_tdma_throughputs, axis=0)
agent_std  = np.std(my_agent_throughputs, axis=0)
aloha_std  = np.std(my_aloha_throughputs, axis=0)
# tdma_std   = np.std(my_tdma_throughputs, axis=0)


### plot
# fig = plt.figure(figsize=(10, 7))
fig = plt.figure()
ax  = fig.add_subplot(111)
agent_line, = plt.plot(agent_mean, color='r', lw=1, label='agent')
agent_optimal_line, = plt.plot(agent_optimal, color='r', lw=3, label='agent optimal')
aloha_line, = plt.plot(aloha_mean, color='b', lw=1, label='aloha')
aloha_optimal_line, = plt.plot(aloha_optimal, color='b', lw=3, label='aloha optimal')
# tdma_lien,  = plt.plot(tdma_mean,  color='g', lw=1, label='tdma')
# tdma_optimal_line,  = plt.plot(tdma_optimal, color='g', lw=3, label='tdma  optimal')
handles, labels = ax.get_legend_handles_labels()
# plt.legend(handles=[agent_line, aloha_line, tdma_lien, agent_optimal_line, aloha_optimal_line, tdma_optimal_line], loc='best')
plt.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5,1), fancybox=True, shadow=True)
plt.fill_between(x, agent_mean-agent_std, agent_mean+agent_std,    
    alpha=0.4, edgecolor='#e50000', facecolor='#e50000',
    linewidth=4, linestyle='dashdot', antialiased=True)
plt.fill_between(x, aloha_mean-aloha_std, aloha_mean+agent_std,    
    alpha=0.4, edgecolor='#0343df', facecolor='#0343df',
    linewidth=4, linestyle='dashdot', antialiased=True)
# plt.fill_between(x, tdma_mean-tdma_std, tdma_mean+tdma_std,    
#     alpha=0.4, edgecolor='#15b01a', facecolor='#15b01a',
#     linewidth=4, linestyle='dashdot', antialiased=True)
plt.grid()
plt.xlabel('Time steps')
plt.ylabel('Throughput')
plt.xlim((0, max_iter))
plt.ylim((0, 0.8))
plt.show()


