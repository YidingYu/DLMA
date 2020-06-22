import numpy as np
from tqdm import tqdm

from environment import ENVIRONMENT
from DQN_brain import DQN

def return_action(action, n_actions=2):
    one_hot_vector = [0] * n_actions
    one_hot_vector[action] = 1
    return one_hot_vector


def return_observation(o):
    if o == 'S':
        return [1, 0, 0, 0]
    elif o == 'F':
        return [0, 1, 0, 0]
    elif o == 'B':
        return [0, 0, 1, 0]
    elif o == 'I':
        return [0, 0, 0, 1]


def main(M, E, F, B, gamma, alpha, idx, max_iter):

    agent_reward_list = []
    tdma_reward_list  = []

    channel_state = env.reset()
    channel_state_length = len(channel_state)
    state = np.zeros((M, channel_state_length))

    print('------------------------------------------')
    print('---------- Start processing ... ----------')
    print('------------------------------------------')

    for i in tqdm(range(max_iter)): 
        state = np.vstack([state[1:], channel_state])
        action = agent.choose_action(state)
        observation, agent_reward, tdma_reward = env.step(action)
        agent_reward_list.append(agent_reward)
        tdma_reward_list.append(tdma_reward)

        # M > 1 
        next_channel_state = return_action(action) + return_observation(observation) + [agent_reward, tdma_reward]
        experience = np.concatenate([channel_state, [action, agent_reward, tdma_reward], next_channel_state])
        # store experience
        agent.add_experience(experience)
        if i > 100 and (i % 5 == 0):
            agent.learn()       # internally iterates default (prediction) model
        channel_state = next_channel_state

    with open(f'rewards/reward1_len{max_iter}_M{M}_E{E}_F{F}_B{B}_gamma{gamma}_alpha{alpha}_idx{idx}.txt', 'w') as my_agent:
        for i in agent_reward_list:
            my_agent.write(str(i) + '   ')
    with open(f'rewards/reward2_len{max_iter}_M{M}_E{E}_F{F}_B{B}_gamma{gamma}_alpha{alpha}_idx{idx}.txt', 'w') as my_tdma:
        for i in tdma_reward_list:
            my_tdma.write(str(i) + '   ') 
        


if __name__ == "__main__":

    n_nodes = 2 # number of nodes
    n_actions = 2 # number of actions

    M = 4 # state length
    E = 1000 # memory size
    F = 20 # target network update frequency
    B = 64 # mini-batch size
    gamma = 0.9 # discount factor

    alpha = 0 # fairness index
    
    max_iter = int(1e4)
    idx = 1


    env = ENVIRONMENT(features=8, 
                      )

    agent = DQN(env.features,
                    n_nodes,
                    n_actions,
                    state_length=M,  
                    memory_size=E,
                    replace_target_iter=F,
                    batch_size=B,
                    gamma=gamma,
                    epsilon=1,
                    epsilon_min=0.005,
                    epsilon_decay=0.995,
                    alpha=alpha
                    )

    main(M, E, F, B, gamma, alpha, idx, max_iter)