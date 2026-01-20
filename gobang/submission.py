from utils import *
import numpy as np
import torch
import torch.nn as nn
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


class Actor(nn.Module):
    """
    The actor is responsible for generating dependable policies to maximize the cumulative reward as much as possible.
    It takes a batch of arrays shaped either (B, 1, N, N) or (N, N) as input, and outputs a tensor shaped (B, N ** 2)
    as the generated policy.
    """

    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size
        """
        # Define your NN structures here. Torch modules have to be registered during the initialization process.
        # For example, you can define CNN structures as follows:

        # self.conv_blocks = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, padding=padding),
        #     nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride),
        #     nn.ReLU(),
        # )

        # Here, channels, kernel_size, padding, and stride are what we would call "Hyperparameters" in deep learning.

        # After convolution, you can flatten (nn.Flatten()) the hidden 2d-representation to obtain the corresponding
        # 1d-representation. Then, fully connected layers can be used to obtain a representation of n**2 dimensions,
        # with each digit indicating the "raw number of policy" (which has to be further constrained and modified
        # in the next step).

        # self.linear_blocks = nn.Sequential(
        #     nn.Linear(in_features=features, out_features=board_size ** 2),
        # )

        # After obtaining a representation of n**2 dimensions, you STILL NEED TO PERFORM ADDITIONAL PROCESSING,
        # including:
        # i) ensuring that all digits corresponding to illegal actions are set to 0 (!!!!!THE MOST IMPORTANT!!!!!);
        # ii) ensuring that the remaining digits satisfy the normalization condition (i.e., the sum of them is equal
        #     to 1).
        # In-place operations are strongly discouraged because they can lead to gradient calculation failures.
        # As an intelligent alternative, consider approaches that can avoid in-place modifications to achieve the goal.

        # You are also encouraged to explore other powerful models and experiment with different techniques,
        # such as using attention modules, different activation functions, or simply adjusting hyperparameter settings.
        """

        # BEGIN YOUR CODE
        hidden_channels = 64
        mid_channels = 96

        # BatchNorm 稳定训练
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=False),
        )

        # 全连接头负责将棋盘级表征映射到 N^2 行为空间
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * board_size * board_size, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, board_size ** 2),
        )
        # END YOUR CODE

        # Define your optimizer here, which is responsible for calculating the gradients and performing optimizations.
        # The learning rate (lr) is another hyperparameter that needs to be determined in advance.
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray):
        if len(x.shape) == 2:
            output = torch.tensor(x).to(device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            output = torch.tensor(x).to(device).to(torch.float32)

        # Further process and transform the data here. Ensure that the output is shaped (B, n ** 2).
        # We have already ensured that the shape of the raw input is unified to be (B, 1, N, N),
        # where B >= 1 represents the number of data in this batch, and N = n is exactly the size of the board.

        # You can continue processing the data here using the modules that were previously registered during the
        # initialization process. For example:

        # output = self.conv_blocks(output)
        # output = nn.Flatten()(output)
        # output = self.linear_blocks(output)

        # And the reminder AGAIN:

        # ****************************************
        # After obtaining a representation of n**2 dimensions, you STILL NEED TO PERFORM ADDITIONAL DATA PROCESSING,
        # including:
        # i) ensuring that all digits corresponding to illegal actions are set to 0 (!!!!!THE MOST IMPORTANT!!!!!);
        # ii) ensuring that the remaining digits satisfy the normalization condition (i.e., the sum of them is equal
        #     to 1).
        # In-place operations are strongly discouraged because they can lead to gradient calculation failures.
        # ****************************************

        # BEGIN YOUR CODE
        state_tensor = output
        features = self.conv_blocks(state_tensor)
        logits = self.policy_head(features)

        # 根据输入状态动态屏蔽非法动作（已有棋子的位置）
        legal_mask = (state_tensor.view(state_tensor.size(0), -1) == 0).to(torch.float32)
        stabilized_logits = logits - logits.max(dim=-1, keepdim=True)[0]
        masked_logits = torch.exp(stabilized_logits) * legal_mask
        denom = masked_logits.sum(dim=-1, keepdim=True)

        # 若棋盘已满导致 denom=0，则回退为均匀分布防止出现 NaN
        uniform_policy = torch.ones_like(masked_logits) / masked_logits.shape[-1]
        normalized_policy = masked_logits / (denom + 1e-8)
        output = torch.where(denom > 0, normalized_policy, uniform_policy)
        # END YOUR CODE
        return output


class Critic(nn.Module):
    """
    The critic is responsible for generating dependable Q-values to fit the solution of Bellman Equations. It takes
    a batch of arrays (shaped either (B, 1, N, N) or (N, N)) and a batch of actions (shaped (B, 2)) as input, and
    outputs a tensor shaped (B, ) as the Q-values on the specified (s, a) pairs.

    For example, actions can be:
    [[0, 1],
     [2, 3],
     [5, 6]]
    which means that there are three actions leading the model to place the pieces on the coordinates (0, 1), (2, 3),
    and (5, 6), respectively. These actions correspond one-to-one with indices 0 * 12 + 1 = 1, 2 * 12 + 3 = 27,
    and 5 * 12 + 6 = 66, assuming n to be 12. You can easily transform a single action to the corresponding digit by
    using _position_to_index, or using _index_to_position vice versa.

    The main idea is that we first obtain a tensor shaped (B, N ** 2) as the Q-values for all possible actions given
    the unified state tensor shaped (B, 1, N, N), and then extract the Q-values corresponding to each action (i, j)
    from the entire Q-value tensor. (_position_to_index should be fully utilized to get the corresponding action indices).
    Finally, it returns a tensor of shape (B,) containing these Q-values.
    """

    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size
        # Define your NN structures here as the same. Torch modules have to be registered during the initialization
        # process.

        # BEGIN YOUR CODE
        critic_channels = 64
        critic_mid = 128

        # 卷积编码器：复用棋盘的局部结构先验，提取长/短连子的空间模式
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, critic_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(critic_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(critic_channels, critic_mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(critic_mid),
            nn.ReLU(inplace=False),
        )

        # 动作价值头：将全局特征映射为 N^2 个动作对应的 Q(s,a)
        self.q_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(critic_mid * board_size * board_size, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, board_size ** 2),
        )
        # END YOUR CODE

        # Define your optimizer here, which is responsible for calculating the gradients and performing optimizations.
        # The learning rate (lr) is another hyperparameter that needs to be determined in advance.
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray, action: np.ndarray):
        indices = torch.tensor([
            _position_to_index(self.board_size, int(x), int(y)) for x, y in action
        ], dtype=torch.long, device=device)
        if len(x.shape) == 2:
            output = torch.tensor(x).to(device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            output = torch.tensor(x).to(device).to(torch.float32)

        # BEGIN YOUR CODE
        state_tensor = output
        features = self.conv_blocks(state_tensor)
        q_map = self.q_head(features)
        q_map = q_map.view(q_map.size(0), -1)

        # 通过索引提取指定动作的 Q 值，保持梯度可回传
        gathered_q = torch.gather(q_map, 1, indices.unsqueeze(-1)).squeeze(-1)
        output = gathered_q
        # END YOUR CODE

        return output


class GobangModel(nn.Module):
    """
    The GobangModel class integrates the Actor and Critic classes for computation and training. Given state tensors "x"
    and action tensors "action", it directly outputs self.actor(x) and self.critic(x, action) as the policy and Q-values
    respectively.
    """

    def __init__(self, board_size: int, bound: int):
        super().__init__()
        self.bound = bound
        self.board_size = board_size

        """
        Register the actor and critic modules here. You do not need to further design the structures at this step.
        Feel free to add extra parameters in the __init__ method of either the Actor class or the Critic class for your 
        convenience, if necessary.
        """

        # BEGIN YOUR CODE
        self.actor = Actor(board_size=board_size)
        self.critic = Critic(board_size=board_size)
        # END YOUR CODE

        self.to(device)

    def forward(self, x, action):
        """
        Return the policy vector π(s) and Q-values Q(s, a) given state "x" and action "action".
        """

        policy = self.actor(x)
        q_values = self.critic(x, action)
        return policy, q_values

    def optimize(self, policy, qs, actions, rewards, next_qs, gamma, eps=1e-6):
        """
        This function calculates the loss for both the actor and critic.
        Using the obtained loss, we can apply optimization algorithms through actor.optimizer and critic.optimizer
        to either maximize the actor's actual objective or minimize the critic's loss.

        There are 3 bugs in the function "optimize" that prevent the model from executing optimizations correctly.
        Identify and debug all errors.
        """

        # detach() 用于阻断梯度传递，防止 next_qs 对 critic 造成影响
        targets = rewards + gamma * next_qs.detach()
        critic_loss = nn.MSELoss()(qs, targets)

        # 将 (x, y) 坐标映射到扁平索引
        actions_long = actions.to(torch.long)
        indices = (actions_long[:, 0] * self.board_size + actions_long[:, 1]).to(device)
        batch_indices = torch.arange(indices.size(0), device=policy.device)
        aimed_policy = policy[batch_indices, indices]
        actor_loss = -torch.mean(torch.log(aimed_policy + eps) * qs.detach())

        # 先清梯度再反传，随后 step 
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        return actor_loss, critic_loss


if __name__ == "__main__":
    if args.use_wandb:
        try:
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
        except ImportError:
            print("Warning: wandb not installed. Install with 'pip install wandb' to enable experiment tracking.")
            print("Continuing without wandb...")
    
    agent = GobangModel(board_size=12, bound=5).to(device)
    train_model(agent, num_episodes=num_episodes, checkpoint=checkpoint)
    
    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
