import torch
from runners.goalbasedrlrunner import GoalBasedRunner
import gymnasium as gym
import gymnasium_robotics
import utils
import hydra
from omegaconf import OmegaConf, open_dict
import wandb
gym.register_envs(gymnasium_robotics)


# Register custom resolver for device detection
OmegaConf.register_new_resolver(
    "cuda_if_available", 
    lambda: "cuda" if torch.cuda.is_available() else "cpu"
)


@hydra.main(version_base=None, config_path='config', config_name='train')
def main(cfg):
    utils.set_seed_everywhere(cfg.seed)
    
    
    
    
    
    env = gym.make(cfg.env)
    eval_env = gym.make(cfg.env, render_mode="rgb_array")
    
    nS = env.observation_space["observation"].shape[0] 
    nG = env.observation_space["desired_goal"].shape[0]
    nA = env.action_space.shape[0]
    obs_dim = nS + nG
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]
    
    # Use open_dict context manager to modify the config
    with open_dict(cfg):
        cfg.nS = nS
        cfg.nA = nA
        cfg.nG = nG
        cfg.agent.obs_dim = obs_dim
        cfg.agent.action_dim = nA
        cfg.agent.action_range = action_range
    
    # DON'T call OmegaConf.resolve() - let Hydra handle lazy resolution
    # OmegaConf.resolve(cfg)  # <-- REMOVE THIS LINE
    
    # Debug: print the config to verify
    # print("="*50)
    # print("Agent config:")
    # print(OmegaConf.to_yaml(cfg.agent))
    # print("="*50)
    run = wandb.init(
        project=cfg.project_name,
        # Group by 'experiment' so the 3 seeds are grouped together in the UI
        group=cfg.experiment, 
        # Give each run a unique name based on its mode and seed
        name=f"{cfg.env}_{cfg.mode}_seed{cfg.seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True, 
        mode="online" if not cfg.debug else "disabled"
    )

    runner = GoalBasedRunner(nS, nA, nG, env, eval_env, env.unwrapped.compute_reward, cfg)
    runner.train()

    # 2. Explicitly finish the run so the next multirun job can start a fresh one
    run.finish()
    

if __name__ == "__main__":
    main()