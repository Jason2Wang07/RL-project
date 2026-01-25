import os
import random
import re
import matplotlib.pyplot as plt
import numpy as np
import copy
from typing import List, Tuple, Union, Optional, Any
from tqdm import tqdm
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    print("Warning: Numba not found. Installing numba (pip install numba) is HIGHLY recommended for H100 training.")
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Current device is {device}.")


@njit(fastmath=True)
def check_win_numba(board, color):
    h, w = board.shape
    for i in range(h):
        for j in range(w - 4):
            if board[i, j] == color and board[i, j+1] == color and \
               board[i, j+2] == color and board[i, j+3] == color and board[i, j+4] == color:
                return True
    for i in range(h - 4):
        for j in range(w):
            if board[i, j] == color and board[i+1, j] == color and \
               board[i+2, j] == color and board[i+3, j] == color and board[i+4, j] == color:
                return True
    for i in range(h - 4):
        for j in range(w - 4):
            if board[i, j] == color and board[i+1, j+1] == color and \
               board[i+2, j+2] == color and board[i+3, j+3] == color and board[i+4, j+4] == color:
                return True
    for i in range(4, h):
        for j in range(w - 4):
            if board[i, j] == color and board[i-1, j+1] == color and \
               board[i-2, j+2] == color and board[i-3, j+3] == color and board[i-4, j+4] == color:
                return True
    return False

@njit
def find_winning_move_numba(board, color, empty_indices):

    for idx in empty_indices:
        r = idx // 12
        c = idx % 12
        board[r, c] = color
        if check_win_numba(board, color):
            board[r, c] = 0
            return idx
        board[r, c] = 0
    return -1


class BatchGobang:
    def __init__(self, num_envs=64, board_size=12):
        self.num_envs = num_envs
        self.board_size = board_size
        self.boards = np.zeros((num_envs, board_size, board_size), dtype=np.int8)
        self.current_players = np.ones(num_envs, dtype=np.int8) 
        self.dones = np.zeros(num_envs, dtype=bool)
        self.winners = np.zeros(num_envs, dtype=np.int8)
        
        self.histories = [[] for _ in range(num_envs)]

    def reset_specific(self, env_indices):
        for i in env_indices:
            self.boards[i].fill(0)
            self.current_players[i] = 1
            self.dones[i] = False
            self.winners[i] = 0
            self.histories[i] = []

    def get_states(self):

        batch_input = self.boards.copy()

        white_turn = (self.current_players == 2)

        if np.any(white_turn):
            batch_input[white_turn] = np.where(
                batch_input[white_turn] == 0, 0, 
                3 - batch_input[white_turn]
            )
            
        return batch_input

    def step(self, actions):

        for i in range(self.num_envs):
            if self.dones[i]: continue
            
            act = actions[i]
            r, c = act // self.board_size, act % self.board_size
            player = self.current_players[i]

            view_board = self.boards[i].copy()
            if player == 2:
                view_board = np.where(view_board==1, 2, np.where(view_board==2, 1, 0))
            
            self.histories[i].append({
                'state': view_board,
                'action': act,
                'player': player
            })

            self.boards[i, r, c] = player

            if check_win_numba(self.boards[i], player):
                self.dones[i] = True
                self.winners[i] = player
            elif np.all(self.boards[i] != 0): 
                self.dones[i] = True
                self.winners[i] = 0 
            
            self.current_players[i] = 3 - player

def augment_data(states, actions, targets, board_size=12):

    aug_s, aug_a, aug_t = [], [], []

    states_np = np.array(states)
    
    for i in range(len(states)):
        s = states_np[i]
        a = actions[i]
        t = targets[i]
        
        x, y = a // board_size, a % board_size
        
        for k in range(4):
            rot_s = np.rot90(s, k)

            dummy = np.zeros((board_size, board_size))
            dummy[x, y] = 1
            rot_dummy = np.rot90(dummy, k)
            rx, ry = np.where(rot_dummy == 1)

            ra = rx[0] * board_size + ry[0]
            
            aug_s.append(rot_s.copy())
            aug_a.append(ra)
            aug_t.append(t)
            
            flip_s = np.fliplr(rot_s)
            flip_dummy = np.fliplr(rot_dummy)
            fx, fy = np.where(flip_dummy == 1)

            fa = fx[0] * board_size + fy[0]
            
            aug_s.append(flip_s.copy())
            aug_a.append(fa)
            aug_t.append(t)
            
    return aug_s, aug_a, aug_t


def train_model(model, num_episodes=50000, checkpoint=1000, resume=None, gamma=0.99):
    print("🚀 Starting High-Performance Batch Training (H100 Optimized)...")
    
    NUM_ENVS = 128          
    BATCH_SIZE = 4096       
    BUFFER_CAPACITY = 20000 
    EPSILON = 0.2           
    
    envs = BatchGobang(num_envs=NUM_ENVS, board_size=model.board_size)
    
    start_ep = 0
    if resume and os.path.isfile(resume):
        print(f"Loading checkpoint {resume}...")
        try:
            model.load_state_dict(torch.load(resume))
            import re
            match = re.search(r'model_(\d+).pth', resume)
            if match: start_ep = int(match.group(1)) + 1
        except Exception as e:
            print(f"Resume failed: {e}")

    buffer_states, buffer_actions, buffer_targets = [], [], []
    
    recent_game_steps = []
    
    pbar = tqdm(range(start_ep, num_episodes))
    
    games_finished = start_ep 
    
    try:
        while games_finished < num_episodes:
            
            states = envs.get_states() 
            
            states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(1).to(device)
            with torch.no_grad():
                probs, _ = model(states_tensor) 
                probs = probs.cpu().numpy()
                
            actions = np.zeros(NUM_ENVS, dtype=np.int64)
            
            for i in range(NUM_ENVS):
                if envs.dones[i]: continue
                
                board = envs.boards[i]
                player = envs.current_players[i]
                opponent = 3 - player

                empty_indices = np.where(board.flatten() == 0)[0]
                if len(empty_indices) == 0:
                    envs.dones[i] = True
                    continue

                win_move = find_winning_move_numba(board, player, empty_indices)
                if win_move != -1:
                    actions[i] = win_move
                    continue
                    
                block_move = find_winning_move_numba(board, opponent, empty_indices)
                if block_move != -1:
                    actions[i] = block_move
                    continue
                    
                p = probs[i]
                p_valid = p[empty_indices]
                if p_valid.sum() > 0:
                    p_valid /= p_valid.sum()

                    if random.random() < EPSILON:
                        move_idx = np.random.choice(len(empty_indices))
                        actions[i] = empty_indices[move_idx]
                    else:
                        move_idx = np.random.choice(len(empty_indices), p=p_valid)
                        actions[i] = empty_indices[move_idx]
                else:
                    actions[i] = np.random.choice(empty_indices)

            envs.step(actions)

            finished_indices = np.where(envs.dones)[0]

            prev_games = games_finished
            
            for idx in finished_indices:
                games_finished += 1
                winner = envs.winners[idx]
                hist = envs.histories[idx]

                recent_game_steps.append(len(hist))

                for step in hist:
                    p = step['player']
                    target = 0.0
                    if winner == p: target = 1.0
                    elif winner != 0: target = -1.0
                    
                    buffer_states.append(step['state'])
                    buffer_actions.append(step['action'])
                    buffer_targets.append(target)

                if games_finished % 10 == 0:
                    pbar.update(10)
                    EPSILON = max(0.05, EPSILON * 0.9999)

            if len(finished_indices) > 0:
                envs.reset_specific(finished_indices)
                
                if games_finished // checkpoint > prev_games // checkpoint:
                     print(f"\nCheckpoint triggered at {games_finished} episodes.")
                     os.makedirs("checkpoints", exist_ok=True)
                     torch.save(model.state_dict(), f"checkpoints/model_{games_finished}.pth")

            if len(buffer_states) >= BUFFER_CAPACITY:

                aug_s, aug_a, aug_t = augment_data(buffer_states, buffer_actions, buffer_targets, model.board_size)

                b_states = torch.tensor(np.array(aug_s), dtype=torch.float32).unsqueeze(1).to(device)
                b_actions = torch.tensor(aug_a, dtype=torch.long).to(device)
                b_targets = torch.tensor(aug_t, dtype=torch.float32).to(device)

                indices = torch.randperm(len(b_states))
                
                model.train()
                total_a_loss = 0
                total_c_loss = 0
                steps = 0

                for start in range(0, len(b_states), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    idx = indices[start:end]
                    
                    mini_s = b_states[idx]
                    mini_a = b_actions[idx]
                    mini_t = b_targets[idx]
                    
                    mini_probs, mini_qs = model(mini_s)

                    a_l, c_l = model.optimize(mini_probs, mini_qs, mini_a, mini_t, None, gamma=0)
                    
                    total_a_loss += a_l
                    total_c_loss += c_l
                    steps += 1

                avg_a = total_a_loss / steps
                avg_c = total_c_loss / steps

                avg_steps = sum(recent_game_steps) / len(recent_game_steps) if recent_game_steps else 0
                
                pbar.set_description(f"Loss A:{avg_a:.3f} C:{avg_c:.3f} Steps:{avg_steps:.1f} Eps:{EPSILON:.2f}")
                
                if WANDB_AVAILABLE:
                    wandb.log({
                        "episode": games_finished,
                        "actor_loss": avg_a,
                        "critic_loss": avg_c,
                        "buffer_size": len(b_states),
                        "avg_game_steps": avg_steps, 
                        "epsilon": EPSILON           
                    })

                buffer_states, buffer_actions, buffer_targets = [], [], []
                recent_game_steps = []

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C). Saving emergency checkpoint...")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_interrupt_{games_finished}.pth")
        print(f"Saved to checkpoints/model_interrupt_{games_finished}.pth")
        
    except Exception as e:
        print(f"\n\nTraining crashed: {e}")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_crash_{games_finished}.pth")
        raise e

    finally:
        pbar.close()
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

class Gobang:
    def __init__(self, board_size, bound, training):
        pass 

def _position_to_index(board_size, x: int, y: int) -> int: return int(x * board_size + y)
def _index_to_position(board_size, index: int) -> Tuple[int, int]: return index // board_size, index - (index // board_size) * board_size

__all__ = ['_position_to_index', '_index_to_position', 'train_model', 'Gobang', 'device']