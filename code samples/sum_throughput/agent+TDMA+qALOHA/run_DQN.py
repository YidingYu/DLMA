from environment import ENVIRONMENT
from DQN_brain import DQN

import numpy as np
import matplotlib.pyplot as plt
import time

def main(max_iter):
    start = time.time()
    agent_reward_list = []
    aloha_reward_list  = []
    tdma_reward_list  = []
    state = env.reset()
    state_length = len(state)
    print('------------------------------------------')
    print('---------- Start processing ... ----------')
    print('------------------------------------------')

    for i in range(max_iter): 
        agent_action = dqn_agent.choose_action(state)
        observation_, reward, agent_reward, aloha_reward, tdma_reward = env.step(agent_action)
        agent_reward_list.append(agent_reward)
        aloha_reward_list.append(aloha_reward)
        tdma_reward_list.append(tdma_reward)
 
        if state_length<3:
            next_state = np.concatenate([agent_action, observation_])
        else:
            next_state = np.concatenate([state[2:], [agent_action, observation_]])

        dqn_agent.store_transition(state, agent_action, reward, next_state)
        if i > 200:
            dqn_agent.learn()       # internally iterates default (prediction) model
        state = next_state

    with open('rewards/agent_len2e5_M20_h6_q0.7_t10-3_3.txt', 'w') as my_agent:
        for i in agent_reward_list:
            my_agent.write(str(i) + '   ')
    with open('rewards/aloha_len2e5_M20_h6_q0.7_t10-3_3.txt', 'w') as my_aloha:
        for i in aloha_reward_list:
            my_aloha.write(str(i) + '   ') 
    with open('rewards/tdma_len2e5_M20_h6_q0.7_t10-3_3.txt', 'w') as my_tdma:
        for i in tdma_reward_list:
            my_tdma.write(str(i) + '   ') 
    # save model 
    # dqn_agent.save_model("models/model_len2e5_M20_h6_q0.6_3.h5")  
    # print the results
    print('-----------------------------')
    print('average agent reward: {}'.format(np.mean(agent_reward_list[-2000:])))
    print('average aloha reward: {}'.format(np.mean(aloha_reward_list[-2000:])))
    print('average tdma  reward: {}'.format(np.mean(tdma_reward_list[-2000:])))
    print('average total reward: {}'.format(np.mean(agent_reward_list[-2000:]) + 
                                            np.mean(aloha_reward_list[-2000:]) +
                                            np.mean(tdma_reward_list[-2000:])))
    print('Time elapsed:', time.time()-start)



if __name__ == "__main__":
    env = ENVIRONMENT(state_size=40, 
                      attempt_prob=0.7
                      )

    dqn_agent = DQN(env.state_size,
                    env.n_actions,  
                    env.n_nodes,
                    memory_size=500,
                    replace_target_iter=200,
                    batch_size=32,
                    learning_rate=0.01,
                    gamma=0.9,
                    epsilon=0.1,
                    epsilon_min=0.005,
                    epsilon_decay=0.995,
                    )

    main(max_iter=200000)