import gymnasium as gym
import gymnasium_robotics
import numpy as np
# Registering once is enough
gym.register_envs(gymnasium_robotics)

env = gym.make('AntMaze_Large-v5')

n_episodes = 5
for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    step = 0
    while not done:
        step+=1
        # Sample random action: torque applied to the ant's joints
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(info)
        done = terminated or truncated
        print(f"step = {step}, reward = {reward}")
        
        # Accessing coordinates if you want to track progress
        # x, y = obs['achieved_goal'][:2] 
        
    print(f"Episode {ep} finished.")

env.close()