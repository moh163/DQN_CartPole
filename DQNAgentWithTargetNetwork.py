import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from cartpole import CustomCartPoleEnv
import matplotlib.pyplot as plt
import time
import os

from gymnasium.wrappers import TimeLimit


device = torch.device("Cuda" if torch.cuda.is_available() else "cpu")
seed = 213
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        #self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(256, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        if hasattr(self, 'fc2'):
            x = self.relu(self.fc2(x))
        if hasattr(self, 'fc3'):
            x = self.relu(self.fc3(x))
        return self.out(x)

class DQNAgentTargetNetwork:
    def __init__(self):
        self.env = TimeLimit(CustomCartPoleEnv(render_mode="rgb_array"), max_episode_steps=500)
        self.set_seed(213) 
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1500
        self.memory = deque(maxlen=5000)
        
        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 128
        self.train_start = 1000

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)  
        self.update_target_model() 
        self.target_update_freq = 500
        self.update_counter = 0

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-4, alpha=0.95, eps=0.01)
        self.criterion = nn.MSELoss()
    def set_seed(self, seed=seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.env.action_space.seed(seed) 


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


    def act(self, state, test=False):
        if not test and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).to(device)  
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = minibatch[i][0]
            actions.append(minibatch[i][1])
            rewards.append(minibatch[i][2])
            next_states[i] = minibatch[i][3]
            dones.append(minibatch[i][4])

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_values = self.model(states)
        q_next = self.target_model(next_states).detach()  

        q_target = q_values.clone().detach()
        for i in range(self.batch_size):
            if dones[i]:
                q_target[i][actions[i]] = rewards[i]
            else:
                q_target[i][actions[i]] = rewards[i] + self.gamma * torch.max(q_next[i])

        loss = self.criterion(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

   
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()

    def save(self, name):
        torch.save(self.model.state_dict(), name)
    
    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=device))
    

    def run(self,render=True, return_rewards=False):
        start_time = time.time()
        episode_rewards = [] 
        recent_scores = deque(maxlen=5)
        for e in range(self.EPISODES):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            cumulative_reward = 0  
            while not done:
                if render:
                    self.env.render()
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.state_size])

                if done and i < self.env._max_episode_steps - 1:
                    reward = -10
                cumulative_reward += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    recent_scores.append(i)
                    episode_rewards.append(cumulative_reward)  
                    if render:
                        print("Episode: {}/{}, score: {}, récompense cumulée: {:.2f}, epsilon: {:.4f}".format(
                            e, self.EPISODES, i, cumulative_reward, self.epsilon))
                    
                    if len(recent_scores) == 5 and np.mean(recent_scores) >=490:
                        total_training_time = time.time() - start_time 
                        print(f"Temps total d'entraînement : {total_training_time:.2f} secondes")
                        print("Saving trained model as cartpole-dqnTargetMoyen.pth")
                        self.save("cartpole-dqnTargetMoyen.pth")
                        if return_rewards:
                            return episode_rewards
                        self.plot_rewards(episode_rewards)
                        return
                self.replay()
        total_training_time = time.time() - start_time  
        print(f"Temps total d'entraînement : {total_training_time:.2f} secondes")
        print("Saving trained model as cartpole-dqnTargetMoyen.pth")
        self.save("cartpole-dqnTargetMoyen.pth")
        if return_rewards:
            return episode_rewards
        self.plot_rewards(episode_rewards)

    def plot_rewards(self, rewards):
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Récompense cumulée")
        plt.title("Récompense cumulée par épisode")
        plt.show()

    def test(self):
        self.load("cartpole-dqnTargetMoyen.pth")
        self.EPISODES = 100
        results = []
        success_count = 0
        episode_rewards = []  
        inference_times = []  
        for e in range(self.EPISODES):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            cumulative_reward = 0  
            start_time = time.time()
            while not done:
                self.env.render()
                action = self.act(state, test=True)
                next_state, reward, terminated, truncated, _= self.env.step(action)
                done = terminated or truncated
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                cumulative_reward += reward
                if done:
                    episode_rewards.append(cumulative_reward) 
                    if i == 500:
                        success = "Oui"
                        success_count += 1
                    else:
                        success = "Non"
                    print("Episode: {}/{}, score: {}, atteint 500: {}".format(e, self.EPISODES, i, success))
                    results.append((e, i, success))
                    break
            inference_time = time.time() - start_time 
            inference_times.append(inference_time)
        print("Nombre d'épisodes réussis (atteignant 500): ", success_count)
        print("Score moyen: ",np.mean(episode_rewards))
        print(f"Temps moyen d'inférence par épisode : {np.mean(inference_times):.4f} secondes")

    def update_optimizer(self, learning_rate):
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate, alpha=0.95, eps=0.01)
    def update_memory_size(self, memory_size):
        self.memory = deque(maxlen=memory_size)


def test_env_param_variations(param_name, values, max_episode_steps=500):
    from cartpole import CustomCartPoleEnv
    from gymnasium.wrappers import TimeLimit
    save_dir = r"C:\Users\Lenovo LEGION\Downloads\DQN_CartPole\params2"
    os.makedirs(save_dir, exist_ok=True)

    results = {}
    all_rewards = {}

    for val in values:
        print(f"\nTesting {param_name} = {val}")
        env_kwargs = {
            param_name: val,
            "render_mode": "rgb_array"
        }
        env = TimeLimit(CustomCartPoleEnv(**env_kwargs), max_episode_steps=max_episode_steps)

        agent = DQNAgentTargetNetwork()
        agent.env = env
        agent.state_size = env.observation_space.shape[0]
        agent.action_size = env.action_space.n

        start_time = time.time()
        episode_rewards = agent.run(render=False, return_rewards=True)
        episodes_taken = len(episode_rewards)
        duration = time.time() - start_time

        print(f"→ Reached requirements in {episodes_taken} episodes ({duration:.2f}s)")

        results[val] = episodes_taken
        all_rewards[val] = episode_rewards

    plt.figure(figsize=(8,5))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title(f"Impact of {param_name} on training speed")
    plt.xlabel(param_name)
    plt.ylabel("Episodes needed")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{param_name}_episodes_plot.png"))

    print(f" Saved plot to {param_name}_episodes_plot.png")
    plt.close()

    for val, rewards in all_rewards.items():
        plt.figure(figsize=(8,5))
        plt.plot(range(len(rewards)), rewards)
        plt.title(f"{param_name}={val} : Cumulative rewards per episode")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{param_name}_{val}_rewards_plot.png"))

        print(f" Saved plot to {param_name}_{val}_rewards_plot.png")
        plt.close()

    plt.figure(figsize=(10,6))
    for val, rewards in all_rewards.items():
        plt.plot(range(len(rewards)), rewards, label=f"{param_name}={val}")

    plt.title(f"Comparaison de {param_name} (cumulative rewards)")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{param_name}_combined_rewards_plot.png"))

    print(f" Saved plot to {param_name}_combined_rewards_plot.png")
    plt.close()



if __name__ == "__main__":
    test_env_param_variations("force_mag", [5.0, 7.5, 10.0, 12.5, 15.0])
    test_env_param_variations("theta_threshold_radians", [0.05, 0.1, 0.15, 0.2,0.4])
    test_env_param_variations("x_threshold", [1,1.5, 2.4, 3.0,4.8])

