import tkinter as tk
from tkinter import messagebox, filedialog
import torch
import numpy as np
import threading
import time
from submission import GobangModel
from utils import device, _index_to_position

class GobangGUI:
    def __init__(self, master, board_size=12):
        self.master = master
        self.master.title("Gobang H100 Battle (AI v.s. Human)")
        self.board_size = board_size
        self.cell_size = 40
        self.offset = 30
        
        # 0: Empty, 1: Black, 2: White
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.model = None
        self.game_over = False
        
        # 玩家设定 (1: Human, 2: AI) 默认人类执黑
        self.human_color = 1 
        self.ai_color = 2
        self.current_turn = 1 # 1 always moves first (Black)

        self.init_ui()

    def init_ui(self):
        # --- 控制面板 ---
        control_frame = tk.Frame(self.master, bg="#f0f0f0")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        btn_style = {"font": ("Arial", 10), "width": 12, "bg": "#ddd"}
        
        self.btn_load = tk.Button(control_frame, text="Load Model", command=self.load_model, **btn_style)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_restart = tk.Button(control_frame, text="Restart", command=self.restart_game, **btn_style)
        self.btn_restart.pack(side=tk.LEFT, padx=5)
        
        self.btn_swap = tk.Button(control_frame, text="Swap Color", command=self.swap_color, **btn_style)
        self.btn_swap.pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(control_frame, text="Load model to start", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.info_label.pack(side=tk.RIGHT, padx=10)

        # --- 胜率显示 ---
        self.winrate_label = tk.Label(self.master, text="AI Win Rate: N/A", font=("Consolas", 10), fg="blue")
        self.winrate_label.pack(side=tk.TOP)

        # --- 棋盘 ---
        canvas_size = self.board_size * self.cell_size + self.offset * 2
        self.canvas = tk.Canvas(self.master, width=canvas_size, height=canvas_size, bg="#E3C088")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)
        
        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        # 坐标轴文字
        for i in range(self.board_size):
            # 横坐标
            x = self.offset + i * self.cell_size
            self.canvas.create_text(x, self.offset - 15, text=str(i), font=("Arial", 8))
            # 纵坐标
            y = self.offset + i * self.cell_size
            self.canvas.create_text(self.offset - 15, y, text=str(i), font=("Arial", 8))

        # 网格线
        for i in range(self.board_size):
            start = self.offset
            end = self.offset + (self.board_size - 1) * self.cell_size
            pos = self.offset + i * self.cell_size
            self.canvas.create_line(start, pos, end, pos) # 横线
            self.canvas.create_line(pos, start, pos, end) # 竖线

        # 天元
        cx = self.offset + (self.board_size // 2) * self.cell_size
        self.canvas.create_oval(cx-3, cx-3, cx+3, cx+3, fill="black")

    def draw_piece(self, row, col, color, is_last=False):
        x = self.offset + col * self.cell_size
        y = self.offset + row * self.cell_size
        r = self.cell_size // 2 - 2
        fill_color = "black" if color == 1 else "white"
        outline = "black" if color == 1 else "#555"
        
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=fill_color, outline=outline)
        
        if is_last:
            # 最新一步画个红点标记
            mark_color = "red" if color == 2 else "white"
            self.canvas.create_rectangle(x-3, y-3, x+3, y+3, fill=mark_color, outline=mark_color)

    def load_model(self):
        file_path = filedialog.askopenfilename(initialdir="./checkpoints", filetypes=[("Model", "*.pth")])
        if file_path:
            try:
                # 重新初始化模型 (确保架构匹配)
                self.model = GobangModel(board_size=self.board_size, bound=5)
                state_dict = torch.load(file_path, map_location=device)
                self.model.load_state_dict(state_dict)
                self.model.to(device)
                self.model.eval()
                self.info_label.config(text="Model Ready!", fg="green")
                self.restart_game()
                print(f"Loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Load failed: {str(e)}\nEnsure submission.py matches the checkpoint.")

    def swap_color(self):
        # 交换执子颜色
        self.human_color, self.ai_color = self.ai_color, self.human_color
        role = "Black (First)" if self.human_color == 1 else "White (Second)"
        messagebox.showinfo("Swap", f"You are now {role}")
        self.restart_game()

    def restart_game(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.game_over = False
        self.current_turn = 1
        self.draw_grid()
        self.winrate_label.config(text="AI Win Rate: N/A")
        
        if not self.model:
            self.info_label.config(text="Please load model")
            return

        if self.human_color == 1:
            self.info_label.config(text="Your Turn (Black)", fg="black")
        else:
            self.info_label.config(text="AI Thinking...", fg="red")
            # AI 先手
            threading.Thread(target=self.ai_move_thread).start()

    def handle_click(self, event):
        if self.game_over or not self.model: return
        if self.current_turn != self.human_color: return # Not your turn

        col = int(round((event.x - self.offset) / self.cell_size))
        row = int(round((event.y - self.offset) / self.cell_size))

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            if self.board[row][col] == 0:
                self.place_piece(row, col, self.human_color)
                
                if not self.game_over:
                    self.info_label.config(text="AI Thinking...", fg="red")
                    threading.Thread(target=self.ai_move_thread).start()

    def place_piece(self, row, col, color):
        self.board[row][col] = color
        self.draw_piece(row, col, color, is_last=True)
        
        # 检查胜利
        if self.check_win(row, col, color):
            winner = "You" if color == self.human_color else "AI"
            color_text = "Black" if color == 1 else "White"
            messagebox.showinfo("Game Over", f"{color_text} ({winner}) Wins!")
            self.game_over = True
            self.info_label.config(text=f"{winner} Wins!", fg="blue")
            return

        # 切换回合
        self.current_turn = 3 - self.current_turn # 1->2, 2->1

    def ai_move_thread(self):
        """AI 思考逻辑 (修复了 Tuple 报错)"""
        time.sleep(0.3) 
        
        # 1. 视角转换
        input_board = self.board.copy()
        if self.ai_color == 2:
            input_board = np.where(self.board == 1, 2, np.where(self.board == 2, 1, 0))
        
        # 2. 转换为 Tensor
        inp = torch.tensor(input_board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # [FIXED] 兼容 tuple 返回值 (probs, value)
            output = self.model.actor(inp)
            if isinstance(output, tuple):
                probs, value = output
                # Value 通常在 [-1, 1] 之间，-1输，1赢
                win_rate = (value.item() + 1) / 2 * 100
            else:
                probs = output
                win_rate = 50.0 # 旧模型没有 value
            
            probs_np = probs.cpu().numpy()[0]
            
            # Mask 非法动作
            valid_mask = (self.board.flatten() == 0)
            probs_np = probs_np * valid_mask
            
            if probs_np.sum() > 0:
                probs_np /= probs_np.sum()
            else:
                # 投降/填空
                pass 

            # Greedy 决策
            action_idx = np.argmax(probs_np)
            r, c = _index_to_position(self.board_size, action_idx)
            
        # 回到主线程更新 UI
        self.master.after(0, lambda: self.ai_move_finish(r, c, win_rate))

    def ai_move_finish(self, row, col, win_rate):
        self.place_piece(row, col, self.ai_color)
        self.winrate_label.config(text=f"AI Win Rate: {win_rate:.1f}%")
        
        if not self.game_over:
            self.info_label.config(text="Your Turn", fg="black")

    def check_win(self, x, y, color):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for step in [1, -1]:
                nx, ny = x, y
                while True:
                    nx += dx * step
                    ny += dy * step
                    if not (0 <= nx < self.board_size and 0 <= ny < self.board_size): break
                    if self.board[nx][ny] == color: count += 1
                    else: break
            if count >= 5: return True
        return False

if __name__ == "__main__":
    root = tk.Tk()
    w, h = 500, 600
    ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()
    x, y = (ws/2) - (w/2), (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    
    app = GobangGUI(root)
    root.mainloop()