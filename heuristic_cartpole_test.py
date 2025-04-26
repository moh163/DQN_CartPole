import math
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from cartpole import CustomCartPoleEnv  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def improved_heuristic_policy(state):
    x, x_dot, theta, theta_dot = state

    angle_weight = 3.0
    angle_velocity_weight = 1.0
    position_weight = 0.5
    velocity_weight = 0.1


    action_score = (angle_weight * theta) + (angle_velocity_weight * theta_dot) + (position_weight * x) + (velocity_weight * x_dot)

    return 1 if action_score > 0 else 0


def run_heuristic_agent():
    env = TimeLimit(CustomCartPoleEnv(render_mode="human"), max_episode_steps=500) 
    episodes = 10  

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            env.render()
            action = improved_heuristic_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1

        print(f" Episode {episode+1} finished after {steps} steps with total reward {total_reward}")

    env.close()

if __name__ == "__main__":
    run_heuristic_agent()
