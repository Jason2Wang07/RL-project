from utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
import sys
import argparse

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--num_episodes', type=int, default=50000, help='number of episodes')
parser.add_argument('--checkpoint', type=int, default=1000, help='the interval of saving models')
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
parser.add_argument('--use_wandb', action='store_true', help='use wandb')
parser.add_argument('--wandb_project', type=str, default='gobang-rl-AI3002', help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
args = parser.parse_args()


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # Apply Attention
        out += residual
        return self.relu(out)

def encode_state(x):
    """
    状态编码: (B, 12, 12) -> (B, 3, 12, 12)
    Channel 0: 己方棋子
    Channel 1: 敌方棋子
    Channel 2: 空位 (提供显式的边界信息)
    """
    if x.ndim == 3:
        x = x.unsqueeze(1)
    player = (x == 1).float()
    opponent = (x == 2).float()
    empty = (x == 0).float()
    return torch.cat([player, opponent, empty], dim=1)

class GobangModel(nn.Module):
    def __init__(self, board_size: int, bound: int):
        super().__init__()
        self.board_size = board_size
        self.bound = bound
        
        self.channels = 128
        self.depth = 12
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            *[ResNetBlock(self.channels) for _ in range(self.depth)]
        )
        
        # Policy Head (Actor)
        self.actor_head = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, board_size ** 2)
        )
        
        # Value Head (Critic)
        self.critic_head = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1) # 输出标量 Value
        )
        
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)

    @property
    def actor(self): return self 
    @property
    def critic(self): return self 

    def forward(self, x):
        if isinstance(x, np.ndarray): x = torch.tensor(x, dtype=torch.float32).to(device)
        if isinstance(x, list): x = torch.tensor(np.array(x), dtype=torch.float32).to(device)
        
        if x.ndim == 2: x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3: x = x.unsqueeze(1)
        
        x_input = encode_state(x)
        
        feat = self.backbone(x_input)
        
        logits = self.actor_head(feat)

        illegal_mask = (x.view(x.size(0), -1) != 0)
        logits = logits.masked_fill(illegal_mask, -1e9)
        probs = F.softmax(logits, dim=-1)
        
        value = self.critic_head(feat)
        
        return probs, value

    def optimize(self, probs, values, actions, rewards, next_values, gamma, eps=1e-8):

        targets = rewards 

        critic_loss = F.smooth_l1_loss(values.squeeze(), targets)

        advantages = (targets - values.squeeze()).detach()

        if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], list):
             indices = torch.tensor([r * self.board_size + c for r, c in actions], dtype=torch.long).to(device)
        elif isinstance(actions, torch.Tensor) and actions.ndim == 2:
             indices = (actions[:, 0] * self.board_size + actions[:, 1]).long()
        else:
             indices = torch.tensor(actions, dtype=torch.long).to(device)

        action_probs = probs.gather(1, indices.unsqueeze(1)).squeeze(1)

        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().mean()

        actor_loss = -(torch.log(action_probs + eps) * advantages).mean()

        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0) 
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

if __name__ == "__main__":
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={
                    "num_episodes": args.num_episodes,
                    "batch_size": 2048,
                    "model": "SE-ResNet-128-H100",
                    "resume": args.resume
                }
            )
        except ImportError:
            pass
    
    agent = GobangModel(board_size=12, bound=5).to(device)
    train_model(agent, num_episodes=args.num_episodes, checkpoint=args.checkpoint, resume=args.resume)
    
    if args.use_wandb and 'wandb' in sys.modules:
        wandb.finish()