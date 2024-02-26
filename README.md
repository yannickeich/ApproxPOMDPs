Code for the AISTATS 2024 paper:

Eich, Yannick, Bastian Alt and Heinz Koeppl. "Approximate Control for Continuous-Time POMDPs". International Conference on Artificial Intelligence and Statistics (AISTATS), 2024.


For an experiment first learn the Q-Function according to the examples in  _examples/MDP_Learning and then do filtering and control like the examples in _examples/Simulation.

Since, Q-Learning can take some time, we provided some pre-learnt Q-functions.

For example run BinomialQueueing_Qfunction.py in _examples/Simulation/Queueing, which approximates the exact filtering distribution by a product binomial distribution and uses a pre-learnt Q-function of the underlying MDP.
