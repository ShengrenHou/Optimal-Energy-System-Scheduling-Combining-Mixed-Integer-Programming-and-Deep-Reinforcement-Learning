
# Optimal Energy System Scheduling Using A Constraint-Aware Reinforcement Learning Algorithm

* This code accompanies the paper <i>Optimal Energy System Scheduling Using A Constraint-Aware Reinforcement Learning Algorithm</i>, to appear in International Journal of Electrical Power & Energy Systems.
# Abstract 
* The massive integration of renewable-based distributed energy resources (DERs) inherently increases the energy system's complexity, especially when it comes to defining its operational schedule. Deep reinforcement learning (DRL) algorithms arise as a promising solution due to their data-driven and model-free features. However, current DRL algorithms fail to enforce rigorous operational constraints (e.g., power balance, ramping up or down constraints) limiting their implementation in real systems. To overcome this, in this paper, a DRL algorithm (namely MIP-DQN) is proposed, capable of \textit{strictly} enforcing all operational constraints in the action space, ensuring the feasibility of the defined schedule in real-time operation. This is done by leveraging recent optimization advances for deep neural networks (DNNs) that allow their representation as a MIP formulation, enabling further consideration of any action space constraints. Comprehensive numerical simulations show that the proposed algorithm outperforms existing state-of-the-art DRL algorithms, obtaining a lower error when compared with the optimal global solution (upper boundary) obtained after solving a mathematical programming formulation with perfect forecast information; while strictly enforcing all operational constraints (even in unseen test days).
# Organization
* Folder "Data" -- Historical and processed data.
* script "agent" and "net"-- General network and agent formulation.
* script "DDPG","SAC","TD3" and "PPO"-- The integration of main process for training, test and plot.
* script "tools"-- General function needed for main process 
* script "random_generator_battery" -- The energy system environment
* Run scripts like DDPG.py after installing all packages. Please have a look for the code structure.
# Dependencies
This code requires installation of the following libraries: ```PYOMO```,```pandas 1.1.4```, ```numpy 1.20.1```, ```matplotlib 3.3.4```, ```pytorch 1.11.0```,  ```math```, you can find more information [at this page](https://ieeexplore.ieee.org/document/9960642).
# Recommended citation
A preprint is available, and you can check this paper for more details  [Link of the paper](https://ieeexplore.ieee.org/document/9960642).
* Paper authors: Hou Shengren, Edgar Mauricio Salazar, Pedro P. Vergara, Peter Palensky
* Accepted for publication at IEEE PES ISGT 2022
* If you use (parts of) this code, please cite the preprint or published paper
## Additional Information 
* I am preparing and organizing the code now
