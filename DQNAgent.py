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
# Définition du modèle DQN en PyTorch
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.out(x)

class DQNAgent:
    def __init__(self):
        self.env = TimeLimit(CustomCartPoleEnv(render_mode="rgb_array"), max_episode_steps=500)
        self.set_seed(55) 
        self.state_size = self.env.observation_space.shape[0]   # CartPole= 4
        self.action_size = self.env.action_space.n               # (gauche, droite)
        self.EPISODES = 250
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95      
        self.epsilon = 1.0     
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000
        
        # Création du modèle et configuration de l'optimiseur (RMSprop avec des hyperparamètres similaires)
        self.model = DQN(self.state_size, self.action_size).to(device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.criterion = nn.MSELoss()
    def set_seed(self, seed=seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.env.action_space.seed(seed) 

    # Stockage d'une transition dans la mémoire
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # ε-greedy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).to(device)  # forme (1, state_size)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())

    #  experience replay sur mini-batch
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        # Préparation des batchs
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
        dones = torch.FloatTensor(dones).to(device)  # 1.0 si terminé, 0.0 sinon
        
        # Calcul des Q-values actuelles et Q-values pour l'état suivant
        q_values = self.model(states)
        q_next = self.model(next_states).detach()  # On ne veut pas rétropropager sur ces valeurs
        
        # Calcul de la cible
        q_target = q_values.clone().detach()
        for i in range(self.batch_size):
            if dones[i]:
                q_target[i][actions[i]] = rewards[i]
            else:
                q_target[i][actions[i]] = rewards[i] + self.gamma * torch.max(q_next[i])
        
        # Calcul de la perte (loss) et optimisation
        loss = self.criterion(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Sauvegarde et chargement du modèle
    def save(self, name):
        torch.save(self.model.state_dict(), name)
    
    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=device))
    
    # Boucle d'entraînement
    def run(self,render=True, return_rewards=False):
        start_time = time.time()
        episode_rewards = []  # Liste pour stocker la récompense cumulée de chaque épisode
        for e in range(self.EPISODES):
            state, _ = self.env.reset(seed=seed)
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            cumulative_reward = 0  # Récompense cumulée pour cet épisode
            while not done:
                if render:
                    self.env.render()
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.state_size])
                # Appliquer une pénalité si l'épisode se termine prématurément
                if done and i < self.env._max_episode_steps - 1:
                    reward = -100
                cumulative_reward += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    episode_rewards.append(cumulative_reward)  # Enregistre la récompense de l'épisode
                    if render:
                        print("Episode: {}/{}, score: {}, récompense cumulée: {:.2f}, epsilon: {:.2f}".format(
                            e, self.EPISODES, i, cumulative_reward, self.epsilon))
                    # Sauvegarde du modèle avec le score max (ici 500)
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.pth")
                        self.save("cartpole-dqn.pth")
                        total_training_time = time.time() - start_time  # Temps total d'entraînement
                        print(f"Temps total d'entraînement : {total_training_time:.2f} secondes")
                        if return_rewards:
                            return episode_rewards
                        self.plot_rewards(episode_rewards)
                        return
                self.replay()
        total_training_time = time.time() - start_time  # Temps total d'entraînement
        print(f"Temps total d'entraînement : {total_training_time:.2f} secondes")
        if return_rewards:
            return episode_rewards
        self.plot_rewards(episode_rewards)

    def plot_rewards(self, rewards):
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Récompense cumulée")
        plt.title("Récompense cumulée par épisode")
        plt.show()
    # Phase de test en chargeant un modèle sauvegardé
    def test(self):
        self.load("cartpole-dqn.pth")
        results = []  # Liste pour stocker les résultats de chaque épisode
        success_count = 0
        episode_rewards = []  # Liste pour stocker la récompense cumulée de chaque épisode
        inference_times = []  # Liste pour stocker le temps d'inférence de chaque épisode
        for e in range(self.EPISODES):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            cumulative_reward = 0  # Récompense cumulée pour cet épisode
            start_time = time.time()
            while not done:
                self.env.render()
                state_tensor = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    action = torch.argmax(self.model(state_tensor)).item()
                next_state, reward, terminated, truncated, _= self.env.step(action)
                done = terminated or truncated
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                cumulative_reward += reward
                if done:
                    episode_rewards.append(cumulative_reward)  # Enregistre la récompense de l'épisode
                    if i == 500:
                        success = "Oui"
                        success_count += 1
                    else:
                        success = "Non"
                    print("Episode: {}/{}, score: {}, atteint 500: {}".format(e, self.EPISODES, i, success))
                    results.append((e, i, success))
                    break
            inference_time = time.time() - start_time  # Temps d'inférence pour cet épisode
            inference_times.append(inference_time)
        print("Nombre d'épisodes réussis (atteignant 500): ", success_count)
        print("Score moyen: ",np.mean(episode_rewards))
        print(f"Temps moyen d'inférence par épisode : {np.mean(inference_times):.4f} secondes")

    def update_optimizer(self, learning_rate):
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate, alpha=0.95, eps=0.01)
    def update_memory_size(self, memory_size):
        self.memory = deque(maxlen=memory_size)

def plot_rl_results(data_dict, title, xlabel, ylabel, window_size=5, save_path=None):
    plt.figure(figsize=(10, 6))
    for label, data in data_dict.items():
        data = np.array(data)
        #smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        episodes = np.arange(len(data))
        plt.plot(episodes, data, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()

def test_hyperparameters(agent_class, hyperparams_dict, episodes=100, output_dir="plotsConstant"):
    for param_name, param_values in hyperparams_dict.items():
        results = {}
        print(f"\n--- Testing variations of '{param_name}' ---")
        for value in param_values:
            label = f"{param_name}={value}"
            print(f"→ {label}")
            agent = agent_class()
            if param_name == 'learning_rate':
                agent.update_optimizer(value)
            elif param_name == 'memory_size':
                agent.update_memory_size(value)
            else:
                setattr(agent, param_name, value)
            agent.EPISODES = episodes
            rewards = agent.run(render=False, return_rewards=True)
            results[label] = rewards

        plot_filename = os.path.join(output_dir, f"{param_name}_comparison.png")
        plot_rl_results(
            results,
            title=f"Effet de {param_name} sur la performance",
            xlabel="Épisodes",
            ylabel="Récompense cumulée",
            save_path=plot_filename
        )


if __name__ == "__main__":
    agent = DQNAgent()
    # Pour entraîner l'agent, décommentez la ligne suivante :
    #agent.run()
    # Pour tester l'agent avec le modèle sauvegardé, utilisez :
    agent.test()
    # Pour tester les hyperparamètres, décommentez les ligne suivante :
    # hyperparams_to_test = {
    #     'learning_rate': [0.0001, 0.00025, 0.001],
    #     'gamma': [0.90, 0.95, 0.99],
    #     'epsilon_decay': [0.995, 0.999],
    #     'batch_size': [32, 64, 128],
    #     'epsilon_min': [0.01, 0.001],
    #     'memory_size': [1000, 2000, 5000] 
    # }

    # test_hyperparameters(DQNAgent, hyperparams_to_test, episodes=1000)
