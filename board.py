import numpy as np

class Board:
    def __init__(self, n=6, n_in_row=4):
        self.n = n
        self.n_in_row = n_in_row
        self.board = np.zeros((n, n), dtype=int)
        self.current_player = 1  # 1: 黑棋, -1: 白棋
        self.last_move = -1
        self.move_count = 0

    def get_legal_moves(self):
        # 返回所有为0的坐标索引 (0 ~ n*n-1)
        return np.where(self.board.flatten() == 0)[0]

    def has_legal_moves(self):
        return len(self.get_legal_moves()) > 0

    def do_move(self, move):
        # 执行落子
        x, y = move // self.n, move % self.n
        self.board[x, y] = self.current_player
        self.last_move = move
        self.move_count += 1
        self.current_player = -self.current_player # 切换玩家

    def check_winner(self):
        # 检查是否有人获胜。返回: 1(黑胜), -1(白胜), 0(平/未分胜负)
        # 注意：这里只检查刚落子的位置即可，为了简单，我们扫描全盘（6x6很快）
        for player in [1, -1]:
            # 横向、纵向、对角线检查
            for i in range(self.n):
                for j in range(self.n):
                    if self.board[i, j] == player:
                        # 横向
                        if j + self.n_in_row <= self.n and np.all(self.board[i, j:j+self.n_in_row] == player):
                            return player
                        # 纵向
                        if i + self.n_in_row <= self.n and np.all(self.board[i:i+self.n_in_row, j] == player):
                            return player
                        # 右下对角
                        if i + self.n_in_row <= self.n and j + self.n_in_row <= self.n and \
                           np.all([self.board[i+k, j+k] == player for k in range(self.n_in_row)]):
                            return player
                        # 左下对角
                        if i + self.n_in_row <= self.n and j - self.n_in_row >= -1 and \
                           np.all([self.board[i+k, j-k] == player for k in range(self.n_in_row)]):
                            return player
        
        if not self.has_legal_moves():
            return 0 # 平局
        return None # 游戏继续

    def get_state(self):
        # 返回适合神经网络输入的格式 (4, 6, 6)
        # 通道0: 当前玩家的子
        # 通道1: 对手的子
        # 通道2: 上一步落子位置 (可选，帮助AI关注最新变化)
        # 通道3: 全1或全0，表示当前是黑还是白 (可选)
        # 为了极简，我们这里只用 (1, 6, 6) 的当前棋盘状态，但在H100版请务必加上历史信息
        return self.board.reshape(1, self.n, self.n) * self.current_player