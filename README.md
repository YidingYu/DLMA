
```
Code samples were posted.
```


## Deep-Reinforcement Learning Multiple Access for Heterogeneous Wireless Networks (DLMA)
Our paper has been accepted by [IEEE JSAC](https://ieeexplore.ieee.org/document/8665952).
### Abstract
This paper investigates a deep reinforcement learning (DRL)-based MAC protocol for heterogeneous wireless networking, referred to as a Deep-reinforcement Learning Multiple Access (DLMA). Specifically, we consider the scenario of a number of networks operating different MAC protocols trying to access the time slots of a common wireless medium. A key challenge in our problem formulation is that we assume our DLMA network does not know the operating principles of the MACs of the other networks—i.e., DLMA does not know how the other MACs make decisions on when to transmit and when not to. The goal of DLMA is to be able to learn an optimal channel access strategy to achieve a certain pre-specified global objective. Possible objectives include maximizing the sum throughput and maximizing alpha-fairness among all networks. The underpinning learning process of DLMA is based on DRL. With proper definitions of the state space, action space, and rewards in DRL, we show that DLMA can easily maximize the sum throughput by judiciously selecting certain time slots to transmit. Maximizing general alpha-fairness, however, is beyond the means of the conventional reinforcement learning (RL) framework. We put forth a new multi-dimensional RL framework that enables DLMA to maximize general alpha-fairness. Our extensive simulation results show that DLMA can maximize sum throughput or achieve proportional fairness (two special classes of alpha-fairness) when coexisting with TDMA and ALOHA MAC protocols without knowing they are TDMA or ALOHA. Importantly, we show the merit of incorporating the use of neural networks into the RL framework (i.e., why DRL and not just traditional RL): specifically, the use of DRL allows DLMA (i) to learn the optimal strategy with much faster speed and (ii) to be more robust in that it can still learn a near-optimal strategy even when the parameters in the RL framework are not optimally set.




