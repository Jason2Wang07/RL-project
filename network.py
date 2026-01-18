import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=6, num_res_blocks=2): # 小规模验证用2层ResBlock
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        
        # 卷积输入层
        self.conv_input = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)
        
        # 残差塔
        self.res_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(num_res_blocks)])
        
        # Policy Head (策略头)
        self.policy_conv = nn.Conv2d(64, 4, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)
        
        # Value Head (价值头)
        self.value_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, 1, 6, 6)
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1) # 输出对数概率
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # 输出 [-1, 1]
        
        return p, v