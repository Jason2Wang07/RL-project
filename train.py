import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import copy
import wandb
import os  # <--- 新增
import pickle # <--- 新增

# 引入你的自定义模块
from board import Board
from mcts import MCTS
from network import AlphaZeroNet

class AlphaZeroTrainer:
    def __init__(self):
        self.board_size = 6
        self.n_in_row = 4
        self.game_batch_num = 5000  # 总训练步数
        self.batch_size = 64
        self.epochs = 5
        self.lr = 0.002
        self.buffer = deque(maxlen=10000)
        self.start_game_idx = 0  # <--- 记录从第几局开始
        self.checkpoint_file = "current_checkpoint.pth" # <--- 存档文件名

        self.board = Board(self.board_size, self.n_in_row)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.policy_value_net = AlphaZeroNet(self.board_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_value_net.parameters(), lr=self.lr, weight_decay=1e-4)

        # <--- 尝试加载断点 --->
        self.load_checkpoint()

        # WandB 初始化
        wandb.init(
            project="alphazero-gobang-6x6",
            # 如果之前的 run id 存在，可以尝试 resume，这里简单起见每次当新 run
            config={
                "board_size": self.board_size,
                "n_in_row": self.n_in_row,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "device": str(self.device),
                "resumed_from": self.start_game_idx
            }
        )

    def save_checkpoint(self, game_idx):
        """保存当前训练状态到硬盘"""
        print(f"Saving checkpoint to {self.checkpoint_file}...")
        checkpoint = {
            'model_state_dict': self.policy_value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'game_idx': game_idx,
            'buffer': self.buffer # 保存经验池，防止重启后“冷启动”
        }
        torch.save(checkpoint, self.checkpoint_file)

    def load_checkpoint(self):
        """尝试从硬盘加载训练状态"""
        if os.path.exists(self.checkpoint_file):
            print(f"Found checkpoint: {self.checkpoint_file}. Loading...")
            try:
                checkpoint = torch.load(self.checkpoint_file, map_location=self.device)
                self.policy_value_net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_game_idx = checkpoint['game_idx'] + 1 # 从下一局开始
                self.buffer = checkpoint['buffer']
                print(f"Successfully resumed from Game {self.start_game_idx}!")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
        else:
            print("No checkpoint found. Starting from scratch.")

    def get_policy_value(self, board_obj):
        board_grid = board_obj.get_state()
        input_tensor = torch.FloatTensor(board_grid).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(input_tensor)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            value = value.item()
            
        return act_probs, value

    def self_play(self, mcts):
        self.board = Board(self.board_size, self.n_in_row)
        mcts.reset()
        states, mcts_probs, current_players = [], [], []
        
        while True:
            temp = 1.0 if self.board.move_count < 6 else 1e-3
            acts, probs = mcts.get_move_probs(self.board, temp)
            
            states.append(self.board.get_state())
            mcts_probs.append(probs)
            current_players.append(self.board.current_player)
            
            move = np.random.choice(len(probs), p=probs)
            self.board.do_move(move)
            
            winner = self.board.check_winner()
            if winner is not None:
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    for i, player in enumerate(current_players):
                        winners_z[i] = 1.0 if winner == player else -1.0
                return winner, zip(states, mcts_probs, winners_z), len(states)
            
            mcts.update_with_move(move)

    def train_step(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs_batch = torch.FloatTensor(np.array(mcts_probs_batch)).to(self.device)
        winner_batch = torch.FloatTensor(np.array(winner_batch)).view(-1, 1).to(self.device)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0

        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            log_act_probs, value = self.policy_value_net(state_batch)
            
            value_loss = F.mse_loss(value, winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_act_probs, 1))
            
            loss = value_loss + policy_loss
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return total_loss / self.epochs, total_policy_loss / self.epochs, total_value_loss / self.epochs

    def run(self):
        mcts = MCTS(self.get_policy_value, c_puct=5, n_playout=400)
        
        # <--- 修改循环：从 start_game_idx 开始，而不是从 0 开始 --->
        for i in range(self.start_game_idx, self.game_batch_num):
            try:
                winner, play_data, game_len = self.self_play(mcts)
                play_data = list(play_data)
                self.buffer.extend(play_data)

                loss, policy_loss, value_loss = 0, 0, 0
                if len(self.buffer) > self.batch_size:
                    loss, policy_loss, value_loss = self.train_step()

                print(f"Game {i+1} | Winner: {winner} | Loss: {loss:.4f} | Steps: {game_len}")
                
                wandb.log({
                    "game": i + 1,
                    "total_loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "game_len": game_len,
                    "winner": winner,
                    "buffer_size": len(self.buffer)
                })

                # <--- 每 50 局自动保存一次 --->
                if (i + 1) % 50 == 0:
                    self.save_checkpoint(i)

            except Exception as e:
                print(f"Error in Game {i+1}: {e}")
                # 遇到错误也尝试紧急保存
                self.save_checkpoint(i)
                raise e

if __name__ == '__main__':
    # --- MCTS Monkey Patch (数值溢出修复版) ---
    def get_move_probs_fixed(self, board, temp=1e-3):
        for _ in range(self._n_playout):
            self._playout(copy.deepcopy(board))
        
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        
        if not act_visits:
            return [], np.zeros(board.n * board.n)

        acts, visits = zip(*act_visits)
        
        # 修复逻辑：温度极低时直接 Argmax
        if temp <= 1e-3:
            probs = [0.0] * len(acts)
            max_visit = max(visits)
            best_indices = [i for i, v in enumerate(visits) if v == max_visit]
            best_idx = np.random.choice(best_indices)
            probs[best_idx] = 1.0
        else:
            try:
                act_probs = list(map(lambda x: x**(1.0/temp), visits))
                prob_sum = sum(act_probs)
                probs = [x / (prob_sum + 1e-10) for x in act_probs]
            except OverflowError:
                probs = [0.0] * len(acts)
                best_idx = np.argmax(visits)
                probs[best_idx] = 1.0
        
        full_probs = np.zeros(board.n * board.n)
        for act, prob in zip(acts, probs):
            full_probs[act] = prob
            
        return acts, full_probs 
    
    MCTS.get_move_probs = get_move_probs_fixed
    
    trainer = AlphaZeroTrainer()
    trainer.run()