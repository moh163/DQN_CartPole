import gymnasium as gym

env = gym.make("CartPole-v1",render_mode="human")

def Random_games():
    for episode in range(10):
        env.reset()
        for t in range(500):
            env.render()
            
            action = env.action_space.sample()

            next_state, reward, terminated, truncated, info = env.step(action)
            
            print(t, next_state, reward, terminated, truncated, info, action)
            if terminated or truncated:
                break
                
Random_games()