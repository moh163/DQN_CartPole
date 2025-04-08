import random
import gymnasium as gym
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("Cuda" if torch.cuda.is_available() else "cpu")

# Définition du modèle DQN en PyTorch
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.out = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.out(x)

class DQNAgent:
    def __init__(self):
        self.env = gym.make("CartPole-v1",render_mode="human")
        self.state_size = self.env.observation_space.shape[0]   # CartPole= 4
        self.action_size = self.env.action_space.n               # (gauche, droite)
        self.EPISODES = 1000
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
    def run(self):
        for e in range(self.EPISODES):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, terminated, truncated, _= self.env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.state_size])
                # Pénalité si l'épisode se termine prématurément
                if done and i < self.env._max_episode_steps - 1:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    print("Episode: {}/{}, score: {}, epsilon: {:.2f}".format(e, self.EPISODES, i, self.epsilon))
                    # Sauvegarde du modele avec le score max (ici 500)
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.pth")
                        self.save("cartpole-dqn.pth")
                        return
                self.replay()
    
    # Phase de test en chargeant un modèle sauvegardé
    def test(self):
        self.load("cartpole-dqn.pth")
        for e in range(self.EPISODES):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                state_tensor = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    action = torch.argmax(self.model(state_tensor)).item()
                next_state, reward, terminated, truncated, _= self.env.step(action)
                done = terminated or truncated
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("Episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

if __name__ == "__main__":
    agent = DQNAgent()
    # Pour entraîner l'agent, décommentez la ligne suivante :
    agent.run()
    # Pour tester l'agent avec le modèle sauvegardé, utilisez :
    #agent.test()
