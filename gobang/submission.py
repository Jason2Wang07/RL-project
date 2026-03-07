from re import X
from utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import *
import sys
import argparse

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--num_episodes', type=int, help='number of episodes')
parser.add_argument('--checkpoint', type=int, help='the interval of saving models')
parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking (requires wandb installed)')
parser.add_argument('--wandb_project', type=str, default='gobang-rl-AI3002', help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
args = parser.parse_args()
num_episodes = args.num_episodes
checkpoint = args.checkpoint
FEATURE_CHANNELS = 144
BASE_KERNEL_SIZE = 5
KERNEL_SIZE = 3
HIDDEN_DEPTH = 12
SE_HIDDEN_DEPTH = 12
SE_REDUCTION = 16
ACTOR_HIDDEN_CHANNELS = 8
CRITIC_FEATURE_CHANNELS = 128
INVALID_ACTION_REWARD = -100


class SEBlock(nn.Module):
	def __init__(self, channels: int, reduction: int):
		super().__init__()
		self.seq = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(1, 3),
			nn.Linear(channels, channels // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channels // reduction, channels, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x: torch.Tensor):
		b, c, _, _ = x.size()
		return x * self.seq(x).view(b, c, 1, 1)


class ResNetBlock(nn.Module):
	def __init__(self, channels: int, kernel_size: int, reduction: int):
		super(ResNetBlock, self).__init__()
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor):
		residual = x
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += residual
		return self.relu(out)

class SEResNetBlock(nn.Module):
	def __init__(self, channels: int, kernel_size: int, reduction: int):
		super(SEResNetBlock, self).__init__()
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)
		self.se = SEBlock(channels, reduction) 
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor):
		residual = x
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out = self.se(out)
		out += residual
		return self.relu(out)


class BackBone(nn.Module):
	"""
	Takes a batch of arrays shaped (B, 1, N, N) as input, and outputs a tensor shaped (B, FEATURE_CHANNELS, N, N)
	"""
	def __init__(self):
		super().__init__()
		self.seq = nn.Sequential(
			nn.Conv2d(3, FEATURE_CHANNELS, BASE_KERNEL_SIZE, padding='same', bias=False),
			nn.BatchNorm2d(FEATURE_CHANNELS),
			nn.ReLU(inplace=True),
			*[SEResNetBlock(FEATURE_CHANNELS, KERNEL_SIZE, SE_REDUCTION) for _ in range(SE_HIDDEN_DEPTH)],
			*[ResNetBlock(FEATURE_CHANNELS, KERNEL_SIZE, SE_REDUCTION) for _ in range(HIDDEN_DEPTH - SE_HIDDEN_DEPTH)],
		)
	
	def forward(self, x: torch.Tensor):
		if x.ndim == 3:
			x = x.unsqueeze(1)
		empty = (x == 0).float()
		player = (x == 1).float()
		opponent = (x == 2).float()
		return self.seq(torch.cat([player, opponent, empty], dim=1))


class Actor(nn.Module):
	"""
	The actor is responsible for generating dependable policies to maximize the cumulative reward as much as possible.
	It takes a batch of arrays shaped either (B, 1, N, N) or (N, N) as input, and outputs a tensor shaped (B, N ** 2)
	as the generated policy.
	"""
	def __init__(self, board_size: int, backbone: nn.Module):
		super().__init__()
		self.board_size = board_size
		object.__setattr__(self, 'backbone', backbone)
		self.head = nn.Sequential(
			nn.Conv2d(FEATURE_CHANNELS, ACTOR_HIDDEN_CHANNELS, kernel_size=1, padding='same'),
			nn.BatchNorm2d(ACTOR_HIDDEN_CHANNELS),
			nn.Flatten(1, 3),
			nn.ReLU(inplace=True),
			nn.Linear(ACTOR_HIDDEN_CHANNELS * board_size ** 2, board_size ** 2),
		)

	def forward(self, x, temp=0.25):
		if isinstance(x, np.ndarray): x = torch.tensor(x, dtype=torch.float32).to(device)
		if isinstance(x, list): x = torch.tensor(np.array(x), dtype=torch.float32).to(device)
		if x.ndim == 2: x = x.unsqueeze(0).unsqueeze(0)
		elif x.ndim == 3: x = x.unsqueeze(1)
		features = self.backbone(x)
		logits = self.head(features) / temp
		illegal_mask = (x.view(x.size(0), -1) != 0)
		logits = logits.masked_fill(illegal_mask, -1e9)
		probs = torch.softmax(logits, dim=1)
		return probs


class Critic(nn.Module):
	def __init__(self, board_size: int, backbone: nn.Module):
		super().__init__()
		self.board_size = board_size
		object.__setattr__(self, 'backbone', backbone)
		self.head = nn.Sequential(
			nn.Conv2d(FEATURE_CHANNELS, ACTOR_HIDDEN_CHANNELS, kernel_size=1, padding='same'),
			nn.BatchNorm2d(ACTOR_HIDDEN_CHANNELS),
			nn.Flatten(1, 3),
			nn.ReLU(inplace=True),
			nn.Linear(ACTOR_HIDDEN_CHANNELS * board_size ** 2, CRITIC_FEATURE_CHANNELS),
			nn.ReLU(inplace=True),
			nn.Linear(CRITIC_FEATURE_CHANNELS, 1),
			nn.Tanh()
		)

	def forward(self, x: np.ndarray, action: Optional[np.ndarray]):
		state = torch.as_tensor(x, device=device, dtype=torch.float32)
		if state.dim() == 2: state = state.unsqueeze(0).unsqueeze(0)
		elif state.dim() == 3: state = state.unsqueeze(1)
		if action is None: return self.head(self.backbone(state))
		actions = torch.as_tensor(action, device=device).view(-1, 2).long()
		b = state.size(0)
		if actions.size(0) != b:
			b = min(b, actions.size(0))
			state = state[:b]
			actions = actions[:b]
		boards = state.squeeze(1).clone()
		i = actions[:, 0].clamp(0, self.board_size - 1)
		j = actions[:, 1].clamp(0, self.board_size - 1)
		occupied = boards[torch.arange(b, device=device), i, j] != 0
		boards[torch.arange(b, device=device), i, j] = torch.where(
			occupied,
			boards[torch.arange(b, device=device), i, j],
			torch.ones(b, device=device, dtype=boards.dtype),
		)
		values = self.head(self.backbone(boards.unsqueeze(1))).view(-1)  # (B,)
		if occupied.any():
			values = values.masked_fill(occupied, float(INVALID_ACTION_REWARD))
		return values


class GobangModel(nn.Module):
	def __init__(self, board_size: int, bound: int):
		super().__init__()
		self.bound = bound
		self.board_size = board_size
		self.backbone = BackBone()
		self.actor = Actor(board_size, self.backbone)
		self.critic = Critic(board_size, self.backbone)
		self.optimizer: optim.Optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

	def forward(self, x, action = None):
		if isinstance(x, np.ndarray): x = torch.tensor(x, dtype=torch.float32).to(device)
		if isinstance(x, list): x = torch.tensor(np.array(x), dtype=torch.float32).to(device)
		if x.ndim == 2: x = x.unsqueeze(0).unsqueeze(0)
		elif x.ndim == 3: x = x.unsqueeze(1)
		if action != None:
			return self.actor(x), self.critic(x, action)
		features = self.backbone(x)
		logits = self.actor.head(features)
		illegal_mask = (x.view(x.size(0), -1) != 0)
		logits = logits.masked_fill(illegal_mask, -1e9)
		probs = torch.softmax(logits, dim=1)
		value = self.critic.head(features)
		return probs, value

	def optimize(self, probs, values, actions, rewards, next_values, gamma, eps=1e-8):
		targets = rewards
		critic_loss = torch.nn.functional.mse_loss(values.squeeze(), targets)
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
		torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
		self.optimizer.step()
		return actor_loss.detach(), critic_loss.detach()


def initialize_weights(model: GobangModel):
	return
	# for name, module in model.named_modules():
	# 	if isinstance(module, (nn.Conv2d, nn.Linear)):
	# 		if 'actor.head' in name and isinstance(module, nn.Linear):
	# 			nn.init.xavier_uniform_(module.weight, 0.05)
	# 			if module.bias is not None:
	# 				nn.init.zeros_(module.bias)
	# 		elif 'critic.head' in name and isinstance(module, nn.Linear):
	# 			nn.init.xavier_uniform_(module.weight, 0.05)
	# 			if module.bias is not None:
	# 				nn.init.zeros_(module.bias)
	# 	elif isinstance(module, nn.BatchNorm2d):
	# 		if module.weight is not None:
	# 			nn.init.ones_(module.weight)
	# 		if module.bias is not None:
	# 			nn.init.zeros_(module.bias)


if __name__ == "__main__":
	import wandb
	wandb.init(
		project=args.wandb_project,
		name=args.wandb_name,
		config={
			"num_episodes": num_episodes,
			"checkpoint": checkpoint,
			"board_size": 12,
			"bound": 5,
		}
	)
	print("Wandb initialized successfully.")