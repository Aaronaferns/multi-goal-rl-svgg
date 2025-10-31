from gymnasium.wrappers import RecordVideo
from algorithm.replay_buffers.memory_with_HER import Buffer
import hydra
import utils
import time
import numpy as np
import wandb
from utils import wandb_log
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

class GoalBasedRunner:
    def __init__(self,nS,nA,nG,env,eval_env,reward_fn, cfg):
        self.env = env
        self.eval_env = eval_env
        self.replay_buffer = Buffer(env,nS,nA,nG,reward_fn,cfg.replay_buffer_capacity, debug=cfg.debug) #has buffer, R, O
        self.goal_setter = hydra.utils.instantiate(cfg.goal_setter, _recursive_ = True)
        print(type(self.goal_setter))
        self.goal_setter.setup_runtime(env, self.replay_buffer)
        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_ = False)
        self.step = 0
        self.cfg = cfg
        self.episode = 0
    
   
        
    def sample_data(self): 
        print("sampling data...")
        avg_traj_len = []
        avg_reward=[]
        num_traj = 0
        while num_traj <self.cfg.num_traj or not self.replay_buffer.full:
            #sample a goal g from q
            g = self.goal_setter.sample_goal()
            
            obs, info = self.env.reset()
            self.agent.reset()
            #change goal:
            self.env.unwrapped.goal = g
            
            #perform rollouts
            done=False
            trajectory_memory=[]
            reward = 0
            traj_len = 0
            while not done:
                
                if not self.replay_buffer.full:
                    act = self.env.action_space.sample()
                else:
                    with utils.eval_mode(self.agent):
                        obs_goal = np.concatenate([obs["observation"], g], axis=0) #goal conditioned
                        act = self.agent.act(obs_goal, sample=True)
                        
               
                if self.replay_buffer.full:
                    self.agent.update(self.replay_buffer, self.step)
                    self.episode+=1
                    if self.episode > 0 and self.episode % self.cfg.eval_frequency == 0:
                            wandb_log('eval/episode', self.episode, self.step)
                            self.eval(self.step)

                obs_, r, terminated, truncated, info =self.env.step(act)
                self.step+=1
                reward += r
                traj_len += 1
                success = info['is_success']
                trajectory_memory.append((g,obs,act,r,obs_,success,info))
                
                if truncated or terminated:
                    done = True   
                obs=obs_  
            
            self.replay_buffer.add(trajectory_memory)
            avg_reward.append(reward)
            avg_traj_len.append(traj_len)
            num_traj+=1
            
        return sum(avg_reward)/len(avg_reward), sum(avg_traj_len)/len(avg_traj_len)
                    
    def train_one_epoch(self):
        
        r, tlen = self.sample_data()
        wandb_log('train/avg_reward_per_epoch', r, self.step)
        wandb_log('train/avg_tlen_per_epoch', tlen, self.step)
        #Train SVGG
        self.goal_setter.train(self.step)
        #improve agent
        # self.agent.update(self.replay_buffer)
        print("training complete")
        
        return r,tlen
    
    def train(self):
        # 
       
        for n in range(self.cfg.num_of_epochs):
           
            print(f"started {n}th epoch")
            r,tlen=self.train_one_epoch()
            print(f"Epoch {n}, Avg reward {r}, Avg Trajectory len {tlen}")
          
            
            
    
    def eval(self, step):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            # Create a fresh environment for each episode
            fresh_env = gym.make(self.cfg.env, render_mode="rgb_array")
            episode_env = RecordVideo(fresh_env, 
                                    video_folder=f"./experiments/{self.cfg.experiment}/videos/{step}/{episode}", 
                                    episode_trigger=lambda e: True)
            
            obs, _ = episode_env.reset()
            self.agent.reset()
            g = obs["desired_goal"]
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    obs_goal = np.concatenate([obs["observation"], g], axis=0)
                    action = self.agent.act(obs_goal, sample=False)
                obs, reward, terminated, truncated, _ = episode_env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    done = True
            episode_env.close()
            fresh_env.close()  # Close the fresh environment too
            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        wandb_log('eval/episode_reward', average_episode_reward, self.step)