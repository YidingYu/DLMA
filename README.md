
```
Code samples were posted.
```


## Deep-Reinforcement Learning Multiple Access for Heterogeneous Wireless Networks (DLMA)
See full version of our paper on [arXiv](https://arxiv.org/pdf/1712.00162.pdf) or our conference version in [IEEE ICC 2018](https://ieeexplore.ieee.org/abstract/document/8422168/).
### Abstract
This paper investigates the use of deep reinforcement learning (DRL) in the design of a "universal" MAC protocol referred to as Deep-reinforcement Learning Multiple Access (DLMA). The design framework is partially inspired by the vision of DARPA SC2, a 3-year competition whereby competitors are to come up with a clean-slate design that "best share spectrum with any network(s), in any environment, without prior knowledge, leveraging on machine-learning technique". While the scope of DARPA SC2 is broad and involves the redesign of PHY, MAC, and Network layers, this paper's focus is narrower and only involves the MAC design. In particular, we consider the problem of sharing time slots among a multiple of time-slotted networks that adopt different MAC protocols. One of the MAC protocols is DLMA. The other two are TDMA and ALOHA. The DRL agents of DLMA do not know that the other two MAC protocols are TDMA and ALOHA. Yet, by a series of observations of the environment, its own actions, and the rewards — in accordance with the DRL algorithmic framework — a DRL agent can learn the optimal MAC strategy for harmonious co-existence with TDMA and ALOHA nodes. In particular, the use of neural networks in DRL (as opposed to traditional reinforcement learning) allows for fast convergence to optimal solutions and robustness against perturbation in hyper-parameter settings, two essential properties for practical deployment of DLMA in real wireless networks.




## Carrier-Sense Multiple Access for Heterogeneous Wireless Networks Using Deep-Reinforcement Learning (CS-DLMA)
The paper introduces carrier sensing and differnt packet lengths into DLMA. This paper has been submitted to a conference on Oct. 15, 2018. See the conference version on [arXiv](https://arxiv.org/abs/1810.06830).  
### Abstract
This paper investigates a new class of carrier-sense multiple access (CSMA) protocols that employ deep reinforcement learning (DRL) techniques for heterogeneous wireless networking, referred to as carrier-sense deep-reinforcement learning multiple access (CS-DLMA). Existing CSMA protocols, such as the medium access control (MAC) of WiFi, are designed for a homogeneous network environment in which all nodes adopt the same protocol. Such protocols suffer from severe performance degradation in a heterogeneous environment where there are nodes adopting other MAC protocols. This paper shows that DRL techniques can be used to design efficient MAC protocols for heterogeneous networking. In particular, in a heterogeneous environment with nodes adopting different MAC protocols (e.g., CS-DLMA, TDMA, and ALOHA), a CS-DLMA node can learn to maximize the sum throughput of all nodes. Furthermore, compared with WiFi's CSMA, CS-DLMA can achieve both higher sum throughput and individual throughputs when coexisting with other MAC protocols. Last but not least, a salient feature of CS-DLMA is that it does not need to know the operating mechanisms of the co-existing MACs. Neither does it need to know the number of nodes using these other MACs.

