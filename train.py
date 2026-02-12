import os
# CRITICAL: These MUST be set before ANY gymnasium or mujoco imports
# This tells MuJoCo to use software rendering instead of looking for a physical monitor.
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import torch
from runners.goalbasedrlrunner import GoalBasedRunner
import gymnasium as gym
import gymnasium_robotics
import utils
import hydra
from omegaconf import OmegaConf, open_dict
import wandb

# Register gymnasium_robotics environments
gym.register_envs(gymnasium_robotics)

# Register custom resolver for device detection if not already present
if not OmegaConf.has_resolver("cuda_if_available"):
    OmegaConf.register_new_resolver(
        "cuda_if_available", 
        lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

class CompatibilityWrapper(gym.Wrapper):
    """
    Ensures 'is_success' key exists in the info dict, 
    mapping from Gymnasium Robotics v5 'success' key.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if 'success' in info:
            info['is_success'] = info['success']
        else:
            # Fallback for safety
            info.setdefault('is_success', False)
        return obs, reward, terminated, truncated, info

@hydra.main(version_base=None, config_path='config', config_name='train')
def main(cfg):
    # Set global seeds for reproducibility
    utils.set_seed_everywhere(cfg.seed)
    
    # Initialize Environments
    # We create them here and ensure they are wrapped for the runner
    train_env = CompatibilityWrapper(gym.make(cfg.env))
    eval_env = CompatibilityWrapper(gym.make(cfg.env, render_mode="rgb_array"))
    
    # Extract environment dimensions
    nS = train_env.observation_space["observation"].shape[0] 
    nG = train_env.observation_space["desired_goal"].shape[0]
    nA = train_env.action_space.shape[0]
    obs_dim = nS + nG
    
    # Action range (standard for MuJoCo is usually [-1, 1])
    action_range = [
        float(train_env.action_space.low.min()),
        float(train_env.action_space.high.max())
    ]
    
    # Inject dynamic environment values back into the Hydra config
    with open_dict(cfg):
        cfg.nS = nS
        cfg.nA = nA
        cfg.nG = nG
        cfg.agent.obs_dim = obs_dim
        cfg.agent.action_dim = nA
        cfg.agent.action_range = action_range
    
    # Start WandB session
    run = wandb.init(
        project=cfg.project_name,
        group=cfg.experiment, 
        name=f"{cfg.env}_{cfg.mode}_seed{cfg.seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True, 
        mode="online" if not cfg.debug else "disabled"
    )

    # Instantiate the runner with the compute_reward function for HER
    runner = GoalBasedRunner(
        nS, nA, nG, 
        train_env, 
        eval_env, 
        train_env.unwrapped.compute_reward, 
        cfg
    )
    
    try:
        # Start training loop
        runner.train()
    except Exception as e:
        print(f"CRITICAL ERROR during training: {e}")
        raise e
    finally:
        # CLEANUP: This is crucial for Sequential Multiruns. 
        # It forces MuJoCo to release the C++ resources and memory.
        print("Cleaning up environments and finishing WandB...")
        train_env.close()
        eval_env.close()
        run.finish()

if __name__ == "__main__":
    main()