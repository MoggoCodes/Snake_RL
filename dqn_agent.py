import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        
        # Process spatial information (snake body, head, food)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Process distance information with 1x1 convolutions
        self.distance_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1),  # 1x1 conv to process distances
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
        )
        
        # Combined features processing
        self.combined_conv = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        # Wider fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),  # Increased width
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_actions)
        )
        
        # Initialize weights with Xavier/Glorot initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def _get_conv_out(self, shape):
        # Process a dummy input to get the conv output size
        spatial_input = torch.zeros(1, 3, shape[0], shape[1])
        distance_input = torch.zeros(1, 2, shape[0], shape[1])
        
        spatial_out = self.spatial_conv(spatial_input)
        distance_out = self.distance_conv(distance_input)
        combined = torch.cat([spatial_out, distance_out], dim=1)
        conv_out = self.combined_conv(combined)
        
        return int(np.prod(conv_out.size()))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Split input into spatial and distance components
        spatial_input = x[:, :3]  # First 3 channels
        distance_input = x[:, 3:]  # Last 2 channels
        
        # Process each path
        spatial_features = self.spatial_conv(spatial_input)
        distance_features = self.distance_conv(distance_input)
        
        # Combine features
        combined = torch.cat([spatial_features, distance_features], dim=1)
        conv_out = self.combined_conv(combined)
        
        # Flatten and pass through FC layers
        return self.fc(conv_out.view(batch_size, -1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, n_actions, device="cpu"):
        # Use MPS (Metal Performance Shaders) if available
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Metal GPU acceleration")
        else:
            self.device = torch.device(device)
            print(f"Using {device}")
        
        self.n_actions = n_actions
        
        # Networks
        self.policy_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.batch_size = 64  # Increased batch size
        self.gamma = 0.99  # Discount factor
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.997  # Slower decay
        self.target_update = 5  # More frequent target updates
        self.epsilon = self.eps_start
        
        # Use Adam optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        # Larger replay buffer
        self.memory = ReplayBuffer(50000)
        
        self.steps = 0
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state)
                if len(state.shape) == 3:
                    state = state.unsqueeze(0)
                state = state.to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
        else:
            return random.randrange(self.n_actions)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from replay buffer
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # Compute Q(s_t, a)
        current_q = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        # Compute max Q(s_{t+1}, a) for all next states
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].detach()
        
        # Compute expected Q values
        expected_q = reward + (1 - done) * self.gamma * next_q
        
        # Compute Huber loss (more stable than MSE)
        loss = nn.SmoothL1Loss()(current_q.squeeze(), expected_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Update epsilon
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps'] 