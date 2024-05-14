import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemoryLSTM
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path, map_location=torch.device('cpu'))

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            ### CODE #### 
            # Choose a random action
            a = torch.tensor([[random.randint(0,self.action_size-1)]], device=device, dtype=torch.long)
        else:
            ### CODE ####
            # Choose the best action
            with torch.no_grad():
                  state = torch.FloatTensor(state).unsqueeze(0).to(device) 
            a = self.policy_net(state).max(1)[1].view(1, 1)
        return a.cpu()

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype = object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).to(device)
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).to(device)
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8)


        # Compute Q(s_t, a), the Q-value of the current state
        ### CODE ####
        state_action_values = self.policy_net(states).gather(1, actions.view(batch_size,-1))
        
        # Compute Q function of next state
        ### CODE ####
        non_final_mask=torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in torch.tensor(next_states) if s is not None]).view(states.size()).to(device)
        next_state_values = torch.zeros(batch_size, device=device).to(device)
        
        
        # Find maximum Q-value of action at next state from policy net
        ### CODE ####
        with torch.no_grad():
            next_state_values[non_final_mask] = self.policy_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.discount_factor) + rewards

        # Compute the Huber Loss
        ### CODE ####
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

