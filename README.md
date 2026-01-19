# Multi-Goal Reinforcement Learning with SVGG & SAC


#### 

An implementation of **Goal-Based Reinforcement Learning** using  
**Soft Actor-Critic (SAC)**, **Hindsight Experience Replay (HER)**, and  
**Stein Variational Goal Generation (SVGG)** for curriculum-style goal sampling.

> This project explores sparse-reward, goal-conditioned environments such as `FetchReach-v4` from `gym-robotics`.  
> It’s part of my ongoing independent research into hierarchical and multi-goal RL systems.

---

## Features
- **Goal-Conditioned SAC** implementation in PyTorch  
- **Hindsight Experience Replay (HER)** buffer for sparse rewards  
- **Stein Variational Goal Generation (SVGG)** for adaptive curriculum learning  
- **Hydra** configuration system for modular experiments  
- **Weights & Biases (W&B)** for experiment tracking and visualizations

---

## Concepts
- **Sparse rewards:** agent only receives a positive reward when it reaches the goal, and -1 otherwise  
- **Curriculum goal sampling:** using SVGG to generate intermediate, learnable goals  
- **Actor-Critic architecture:** following SAC’s entropy-regularized policy learning  
- **Replay augmentation:** via Hindsight Experience Replay to reuse failed episodes

---

## Environment
- `gym-robotics`
---

## Setup

```bash
git clone https://github.com/Aaronaferns/multi-goal-rl-svgg.git
cd multi-goal-rl-svgg
pip install -r requirements.txt
python train.py experiment={any name of your choice}
