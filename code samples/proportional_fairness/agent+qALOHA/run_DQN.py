import numpy as np

from environment import ENVIRONMENT
from DQN_brain import DQN
import matplotlib.pyplot as plt
import time

def main(max_iter):
    start = time.time()
    agent_reward_list = []
    aloha_reward_list  = []
    state = env.reset()
    print('------------------------------------------')
    print('---------- Start processing ... ----------')
    print('------------------------------------------')

    for i in range(max_iter): 
        agent_action = dqn_agent.choose_action(state)
        observation_, reward, agent_reward, aloha_reward = env.step(agent_action)
        agent_reward_list.append(agent_reward)
        aloha_reward_list.append(aloha_reward)
 
        if len(state)<5:
            next_state = np.concatenate([agent_action, observation_, agent_reward, aloha_reward])
        else:
            next_state = np.concatenate([state[4:], [agent_action, observation_, agent_reward, aloha_reward]])

        dqn_agent.store_transition(state, agent_action, agent_reward, aloha_reward, next_state)
        if i > 200:
            dqn_agent.learn()       # internally iterates default (prediction) model
        state = next_state

    with open('rewards/agent_len1e5_M20_h6_q0.8_1.txt', 'w') as my_agent:
        for i in agent_reward_list:
            my_agent.write(str(i) + '   ')
    with open('rewards/aloha_len1e5_M20_h6_q0.8_1.txt', 'w') as my_aloha:
        for i in aloha_reward_list:
            my_aloha.write(str(i) + '   ') 
    # save model 
    # dqn_agent.save_model("models/model_len1e5_M20_h6_q0.1_1.h5")  
    # print the results
    print('-----------------------------')
    print('average agent reward: {}'.format(np.mean(agent_reward_list[-2000:])))
    print('average aloha reward: {}'.format(np.mean(aloha_reward_list[-2000:])))
    print('average total reward: {}'.format(np.mean(agent_reward_list[-2000:]) + 
                                            np.mean(aloha_reward_list[-2000:])))
    print('Time elapsed:', time.time()-start)



if __name__ == "__main__":
    env = ENVIRONMENT(state_size=80, 
                      attempt_prob=0.8
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

    main(max_iter=100000)