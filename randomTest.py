import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

def Random_games():
    for episode in range(10):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            env.render()
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        print(f"Episode {episode+1} finished after {steps} steps with total reward {total_reward}")

Random_games()
env.close()
