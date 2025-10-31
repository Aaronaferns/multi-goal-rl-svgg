import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchPickAndPlace-v4");
# env = RecordVideo(env, video_folder="./videos/1", episode_trigger=lambda e: True)
# env = gym.make("FetchPickAndPlaceDense-v4")
goal = np.array([1.0, 2.0, 3.0])
obs = env.reset()
env.unwrapped.goal = goal



n_episodes = 20
for ep in range(n_episodes):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(reward)
env.close()
# video_folder = "./videos/1"
# # Collect all mp4 files sorted by filename
# video_files = sorted([
#     os.path.join(video_folder, f) 
#     for f in os.listdir(video_folder) 
#     if f.endswith(".mp4")
# ])

# Load all videos as VideoFileClip objects
# clips = [VideoFileClip(f) for f in video_files]

# # Concatenate into a single video
# final_clip = concatenate_videoclips(clips)

# # Save the combined video
# final_clip.write_videofile("./videos/combined_eval.mp4", codec="libx264")

# # Close clips to release resources
# for clip in clips:
#     clip.close()
# final_clip.close()
# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# # The following always has to hold:
# assert reward == env.unwrapped.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
# assert truncated == env.unwrapped.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
# assert terminated == env.unwrapped.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# # However goals can also be substituted:
# substitute_goal = obs["achieved_goal"].copy()
# substitute_reward = env.unwrapped.compute_reward(obs["achieved_goal"], substitute_goal, info)
# substitute_terminated = env.unwrapped.compute_terminated(obs["achieved_goal"], substitute_goal, info)
# substitute_truncated = env.unwrapped.compute_truncated(obs["achieved_goal"], substitute_goal, info)

# print(obs["desired_goal"])
# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# print(float(env.action_space.low.min()))