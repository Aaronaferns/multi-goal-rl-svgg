import os
import numpy as np
import torch
import hydra
import wandb
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo

# Internal imports
import utils
from utils import wandb_log
from algorithm.replay_buffers.memory_with_HER import Buffer

# Ensure environments are registered
gym.register_envs(gymnasium_robotics)

class GoalBasedRunner:
    def __init__(self, nS, nA, nG, env, eval_env, reward_fn, cfg):
        self.cfg = cfg
        self.env = env
        
        # PERSISTENT EVAL ENV: Initializing once prevents recursive C++ initialization errors
        # We don't wrap it in RecordVideo yet because we want unique folders per eval call
        self.base_eval_env = eval_env
        
        self.replay_buffer = Buffer(
            env, nS, nA, nG, reward_fn, 
            cfg.replay_buffer_capacity, 
            useHindsight=cfg.use_hindsight, 
            debug=cfg.debug
        )
        
        if cfg.use_goal_setter: 
            self.goal_setter = hydra.utils.instantiate(cfg.goal_setter, _recursive_=True)
            self.goal_setter.setup_runtime(env, self.replay_buffer)
        
        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False)
        
        self.step = 0
        self.episode = 0
        self.checkpoint_dir = cfg.checkpoint_dir
        self.checkpoint_frequency = cfg.checkpoint_frequency
        self.save_final = cfg.save_final

    def save_checkpoint(self, epoch):
        """Save agent and goal_setter state."""
        path = os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch}.pt")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        state = {
            "epoch": epoch, 
            "step": self.step,
            "agent": self.agent.state_dict() if hasattr(self.agent, "state_dict") else None
        }
        
        if self.cfg.use_goal_setter and hasattr(self.goal_setter, "state_dict"):
            state["goal_setter"] = self.goal_setter.state_dict()
            
        torch.save(state, path)
        last_path = os.path.join(self.checkpoint_dir, "last.pt")
        torch.save(state, last_path)
        
        if wandb.run is not None:
            wandb.save(path)
            wandb.save(last_path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path=None):
        load_path = path or os.path.join(self.checkpoint_dir, "last.pt")
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"No checkpoint at {load_path}")
            
        state = torch.load(load_path, map_location="cpu")
        if state["agent"] and hasattr(self.agent, "load_state_dict"):
            self.agent.load_state_dict(state["agent"])
        if "goal_setter" in state and self.cfg.use_goal_setter:
            self.goal_setter.load_state_dict(state["goal_setter"])
            
        self.step = state.get("step", self.step)
        return state.get("epoch", -1)

    def sample_data(self): 
        print(f"sampling data... (Buffer: {self.replay_buffer.size}/{self.cfg.replay_buffer_capacity})")
        avg_traj_len = []
        avg_reward = []
        num_traj = 0
        
        while num_traj < self.cfg.num_traj or not self.replay_buffer.full:
            obs, info = self.env.reset()
            self.agent.reset()
            
            if self.cfg.use_goal_setter:
                g = self.goal_setter.sample_goal(obs)
                self.env.unwrapped.goal = g
            else:
                g = obs["desired_goal"]
            
            done = False
            trajectory_memory = []
            reward = 0
            traj_len = 0
            
            while not done:
                # Decide Action
                if not self.replay_buffer.full:
                    act = self.env.action_space.sample()
                else:
                    with utils.eval_mode(self.agent):
                        obs_goal = np.concatenate([obs["observation"], g], axis=0)
                        act = self.agent.act(obs_goal, sample=True)
                
                # Update agent if buffer has enough data
                if self.replay_buffer.full:
                    self.agent.update(self.replay_buffer, self.step)
                    self.episode += 1
                    if self.episode % self.cfg.eval_frequency == 0:
                        wandb.log({'eval/episode_count': self.episode}, step=self.step)
                        self.eval(self.step)

                # Env Step
                obs_, r, terminated, truncated, info = self.env.step(act)
                self.step += 1
                reward += r
                traj_len += 1
                
                # Success is often in 'is_success', verify your wrapper matches this
                success = info.get('is_success', info.get('success', False))
                trajectory_memory.append((g, obs, act, r, obs_, success, info))
                
                if truncated or terminated:
                    done = True 
                obs = obs_  
            
            self.replay_buffer.add(trajectory_memory)
            avg_reward.append(reward)
            avg_traj_len.append(traj_len)
            num_traj += 1
            
        return np.mean(avg_reward), np.mean(avg_traj_len)
                    
    def train_one_epoch(self):
        r, tlen = self.sample_data()
        wandb.log({
            'train/avg_reward_per_epoch': r,
            'train/avg_tlen_per_epoch': tlen
        }, step=self.step)
        
        if self.cfg.use_goal_setter: 
            self.goal_setter.train(self.step)
            
        self.agent.update(self.replay_buffer, self.step)
        print("Epoch training complete")
        return r, tlen
    
    def train(self):
        for n in range(self.cfg.num_of_epochs):
            print(f"--- Starting Epoch {n} ---")
            r, tlen = self.train_one_epoch()
            
            do_checkpoint = (
                (self.checkpoint_frequency > 0 and (n + 1) % self.checkpoint_frequency == 0)
                or (self.save_final and n == self.cfg.num_of_epochs - 1)
            )
            if do_checkpoint:
                self.save_checkpoint(n)

    def eval(self, step):
        """Perform evaluation and save video."""
        print(f"Running evaluation at step {step}...")
        average_episode_reward = 0
        
        # UNIQUE VIDEO PATH: prevents parallel jobs from overwriting each other
        # Structure: experiments/exp_name/videos/mode_seed/step_number
        video_path = os.path.join(
            "experiments", 
            self.cfg.experiment, 
            "videos", 
            f"{self.cfg.mode}_seed{self.cfg.seed}", 
            str(step)
        )
        
        # Temporarily wrap for recording
        eval_env = RecordVideo(
            self.base_eval_env, 
            video_folder=video_path, 
            episode_trigger=lambda e: True,
            disable_logger=True
        )

        for episode in range(self.cfg.num_eval_episodes):
            obs, _ = eval_env.reset()
            self.agent.reset()
            g = obs["desired_goal"]
            done = False
            episode_reward = 0
            
            while not done:
                with utils.eval_mode(self.agent):
                    obs_goal = np.concatenate([obs["observation"], g], axis=0)
                    action = self.agent.act(obs_goal, sample=False)
                
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
            average_episode_reward += episode_reward
        
        # Important: close the wrapper to flush the video to disk
        eval_env.close()
        
        avg_reward = average_episode_reward / self.cfg.num_eval_episodes
        wandb.log({'eval/avg_reward': avg_reward}, step=step)
        print(f"Eval complete. Avg Reward: {avg_reward}")