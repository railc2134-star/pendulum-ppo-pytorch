Pendulum Continuous Control Agent (Actor-Critic)

This project implements a reinforcement learning agent that learns to solve the Pendulum-v1 environment using an actor-critic method inspired by PPO (Proximal Policy Optimization).

The agent learns to control a continuous action space using stochastic policy gradients.

Environment

- Gymnasium Pendulum-v1
- Continuous action space
- Goal: keep pendulum balanced with minimal energy

Model Architecture

Actor Network:
- Fully connected neural network
- Outputs mean and standard deviation of a Gaussian policy
- Samples continuous actions from Normal distribution

Critic Network:
- Fully connected neural network
- Estimates state-value function V(s)

Training Method

The agent is trained using:
- On-policy rollout collection
- Monte Carlo returns
- Advantage estimation
- Policy gradient updates with clipping (PPO-inspired)
- Value function regression

Key Concepts

- Actor-Critic learning
- Policy gradient methods
- Advantage normalization
- Entropy regularization for exploration
- Gaussian stochastic policy

Limitations

This implementation is simplified and may differ from full PPO:

- No GAE (Generalized Advantage Estimation)
- No proper PPO ratio clipping range
- No minibatch training from rollout buffer
- No value loss coefficient balancing
- No learning rate scheduling
