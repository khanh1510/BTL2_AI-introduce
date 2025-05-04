import tkinter as tk
from tkinter import messagebox
import time, psutil, os
import random
import csv

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),         (0, 1),
              (1, -1),  (1, 0), (1, 1)]
class Board(list):
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    def __init__(self, size=8):
        # Khởi tạo danh sách hai chiều (mảng 2D)
        self.size = size
        super().__init__([[self.EMPTY for _ in range(size)] for _ in range(size)])
        self.initialize()

    def initialize(self):
        mid = self.size // 2
        self[mid - 1][mid - 1] = self.WHITE
        self[mid][mid] = self.WHITE
        self[mid - 1][mid] = self.BLACK
        self[mid][mid - 1] = self.BLACK

    def get_valid_moves(self, player):
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.is_valid_move((r, c), player)]

    def is_valid_move(self, index, player):
        row, col = index
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self[row][col] != self.EMPTY:
            return False
        opponent = -player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            has_opponent = False
            while 0 <= r < self.size and 0 <= c < self.size:
                if self[r][c] == opponent:
                    has_opponent = True
                elif self[r][c] == player:
                    if has_opponent:
                        return True
                    break
                else:
                    break
                r += dr
                c += dc
        return False

    def make_move(self, index, player):
        row, col = index
        if not self.is_valid_move(index, player):
            return False
        self[row][col] = player
        opponent = -player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            pieces_to_flip = []
            while 0 <= r < self.size and 0 <= c < self.size:
                if self[r][c] == opponent:
                    pieces_to_flip.append((r, c))
                elif self[r][c] == player:
                    for rr, cc in pieces_to_flip:
                        self[rr][cc] = player
                    break
                else:
                    break
                r += dr
                c += dc
        return True

    def count_score(self):
        black = sum(cell == self.BLACK for row in self for cell in row)
        white = sum(cell == self.WHITE for row in self for cell in row)
        return black, white

    def is_terminal(self):
        return not self.get_valid_moves(self.BLACK) and not self.get_valid_moves(self.WHITE)

    def copy(self):
        new_board = Board(self.size)
        new_board[:] = [row[:] for row in self]
        return new_board

    def flipped_count(self, move, player):
        flipped = 0
        directions = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]
        x, y = move
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            temp_flipped = 0
            while 0 <= nx < 8 and 0 <= ny < 8 and self[nx][ny] == -player:
                temp_flipped += 1
                nx += dx
                ny += dy
            if 0 <= nx < 8 and 0 <= ny < 8 and self[nx][ny] == player:
                flipped += temp_flipped
        return flipped

    def __repr__(self):
        symbols = {self.EMPTY: '.', self.BLACK: 'B', self.WHITE: 'W'}
        board_str = "  " + " ".join(str(i) for i in range(self.size)) + "\n"
        for i, row in enumerate(self):
            board_str += str(i) + " " + " ".join(symbols[cell] for cell in row) + "\n"
        return board_str

def evaluate(board, player):
    weights = [
        [100, -10,  11,   6,   6,  11, -10, 100],
        [-10, -20,   1,   2,   2,   1, -20, -10],
        [ 10,   1,   5,   4,   4,   5,   1,  10],
        [  6,   2,   4,   2,   2,   4,   2,   6],
        [  6,   2,   4,   2,   2,   4,   2,   6],
        [ 10,   1,   5,   4,   4,   5,   1,  10],
        [-10, -20,   1,   2,   2,   1, -20, -10],
        [100, -10,  11,   6,   6,  11, -10, 100],
    ]

    black, white = board.count_score()
    # Xác định giai đoạn trận đấu
    # Giai đoạn early (0–12 quân): ưu tiên kiểm soát giữa bàn.
    # Giai đoạn mid (13–50 quân): ưu tiên di chuyển linh hoạt (mobility).
    # Giai đoạn late (sắp hết bàn): ưu tiên ăn nhiều quân (greedy).
    total_discs = black + white    
    if total_discs <= 12:
        phase = 'early'
    elif total_discs <= 50:
        phase = 'mid'
    else:
        phase = 'late'

    pos_score = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] == player:
                pos_score += weights[i][j]
            elif board[i][j] == -player:
                pos_score -= weights[i][j]



    my_moves = board.get_valid_moves(player)
    opp_moves = board.get_valid_moves(-player)

    if phase == 'early':
        mobility_weight = 0.2
    elif phase == 'mid':
        mobility_weight = 1.0
    else:
        mobility_weight = 0.5

    mobility_score = mobility_weight * (len(my_moves) - len(opp_moves))



    greedy_score = 0
    for move in my_moves:
        greedy_score += board.flipped_count(move, player)
    greedy_score *= 0.5


    corners = [(0,0), (0,7), (7,0), (7,7)]
    corner_score = 0
    for i,j in corners:
        if board[i][j] == player:
            corner_score += 25
        elif board[i][j] == -player:
            corner_score -= 25


    total_score = pos_score + mobility_score + greedy_score + corner_score
    return total_score

def minimax(board, depth, player, maximizing, alpha=float("-inf"), beta=float("inf"), step=None):
    if step is not None:
        step[0] += 1

    if depth == 0 or board.is_terminal():
        return evaluate(board, player), None

    valid_moves = board.get_valid_moves(player)
    if not valid_moves:
        return minimax(board, depth - 1, -player, not maximizing, alpha, beta, step)

    best_move = None
    if maximizing:
        max_eval = float("-inf")
        for move in valid_moves:
            new_board = board.copy()
            new_board.make_move(move, player)
            eval, _ = minimax(new_board, depth - 1, -player, False, alpha, beta, step)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in valid_moves:
            new_board = board.copy()
            new_board.make_move(move, player)
            eval, _ = minimax(new_board, depth - 1, -player, True, alpha, beta, step)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

   
class GameApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("600x700")
        self.root.configure(bg="#2e3440")        
        self.mode = tk.StringVar(value="PvP")  # Biến lưu chế độ chơi
        self.player = tk.IntVar(value=1)  # Biến lưu lựa chọn player (1 hoặc 2)
        self.depth = tk.IntVar(value=1)
        self.depth1 = tk.IntVar(value=1)  # Biến lưu depth cho AI1
        self.depth2 = tk.IntVar(value=5)  # Biến lưu depth cho AI2
        self.seed = tk.IntVar(value=42)  # Biến lưu seed (dành cho TestAI)
        
        
        self.setup_start_screen()

    def setup_start_screen(self):
        self.clear_root()
        # Tạo nhãn cho chế độ chơi
        tk.Label(self.root, text="Chọn chế độ chơi:", font=("Helvetica", 14)).pack(pady=10)

        # Tạo lựa chọn chế độ chơi
        mode_options = ["TestAI", "PvP", "PvAI", "AIvAI"]
        self.mode_menu = tk.OptionMenu(self.root, self.mode, *mode_options, command=self.on_mode_change)
        self.mode_menu.pack(pady=5)

        # Các lựa chọn bổ sung tùy theo chế độ
        self.additional_options_frame = tk.Frame(self.root)
        self.additional_options_frame.pack()

        # Nút bắt đầu trò chơi
        tk.Button(self.root, text="Bắt đầu", font=("Helvetica", 14), command=self.start_game).pack(pady=20)

        self.on_mode_change(self.mode.get())  # Đảm bảo giao diện được cập nhật ngay khi bắt đầu


    def on_mode_change(self, mode):
        """Thay đổi giao diện khi người dùng chọn chế độ khác."""
        # Ẩn tất cả lựa chọn thêm trước
        for widget in self.additional_options_frame.winfo_children():
            widget.pack_forget()

        if mode == "PvP":
            # Không có lựa chọn thêm cho PvP
            pass
        elif mode == "PvAI":
            # Chọn lượt đi Player (1 hoặc 2)
            tk.Label(self.additional_options_frame, text="Chọn lượt đi của player (1 hoặc 2):", font=("Helvetica", 12)).pack(pady=5)
            player_options = [1, 2]
            tk.OptionMenu(self.additional_options_frame, self.player, *player_options).pack(pady=5)

            tk.Label(self.additional_options_frame, text="Chọn depth (1 đến 10):", font=("Helvetica", 12)).pack(pady=5)
            tk.Scale(self.additional_options_frame, from_=1, to=10, orient="horizontal", variable=self.depth).pack(pady=10)

        elif mode == "AIvAI":
            # Chọn depth cho cả 2 AI
            tk.Label(self.additional_options_frame, text="Chọn depth cho AI 1 (1 đến 10):", font=("Helvetica", 12)).pack(pady=5)
            tk.Scale(self.additional_options_frame, from_=1, to=10, orient="horizontal", variable=self.depth1).pack(pady=10)

            tk.Label(self.additional_options_frame, text="Chọn depth cho AI 2 (1 đến 10):", font=("Helvetica", 12)).pack(pady=5)
            tk.Scale(self.additional_options_frame, from_=1, to=10, orient="horizontal", variable=self.depth2).pack(pady=10)

        elif mode == "TestAI":
            tk.Label(self.additional_options_frame, text="Chọn lượt đi AI test (1 hoặc 2):", font=("Helvetica", 12)).pack(pady=5)
            player_options = [1, 2]
            tk.OptionMenu(self.additional_options_frame, self.player, *player_options).pack(pady=5)
            
            tk.Label(self.additional_options_frame, text="Chọn depth cho AI (1 đến 10):", font=("Helvetica", 12)).pack(pady=5)
            tk.Scale(self.additional_options_frame, from_=1, to=10, orient="horizontal", variable=self.depth).pack(pady=10)

            tk.Label(self.additional_options_frame, text="Chọn seed:", font=("Helvetica", 12)).pack(pady=5)
            tk.Entry(self.additional_options_frame, textvariable=self.seed).pack(pady=10)

    def start_game(self):
        self.clear_root()
        mode = self.mode.get()
        player = self.player.get()
        depth = self.depth.get()
        depth1 = self.depth1.get()
        depth2 = self.depth2.get()
        seed = self.seed.get()
        
        self.game = OthelloGUI(root=self.root, mode=mode,
                               player=player, depth=depth, depth1=depth1, depth2=depth2, 
                               seed=seed
                               )

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()


class OthelloGUI:
    def __init__(self, root, size=8, mode="PvP", player = 1, depth = 5, depth1 = 5, depth2 = 5, seed = 42):
        self.root = root
        self.size = size
        self.mode = mode
        self.player = player
        self.depth = depth
        self.depth1 = depth1
        self.depth2 = depth2
        self.seed = seed
        
        self.board = Board(size)
        self.current_player = Board.BLACK
        self.cell_size = 60

        self.score_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#2e3440", fg="white")
        self.score_label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=size * self.cell_size, height=size * self.cell_size, bg="#3b4252", bd=0, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.player_click)



        self.update_UI()
        if self.mode == "PvP":
            self.player_vs_player()
        elif self.mode == "PvAI":
            self.player_vs_agent(player=self.player, depth=self.depth)
        elif self.mode == "AIvAI":
            self.agent_vs_agent(depth1=self.depth1, depth2=self.depth2)
        elif self.mode == "TestAI":
            self.test_agent(player=self.player, depth=self.depth)





        self.total_step = 0
        self.total_score = 0
        
        
    def update_UI(self):
        self.canvas.delete("all")
        
        # Vẽ lại bàn cờ
        for r in range(self.board.size):
            for c in range(self.board.size):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#4c566a", outline="#2e3440", width=2)

                if self.board[r][c] == Board.BLACK:
                    self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="black", outline="")
                elif self.board[r][c] == Board.WHITE:
                    self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="white", outline="")

        black, white = self.board.count_score()
        self.score_label.config(text=f"Đen: {black} | Trắng: {white}")
        
        # Lấy các nước đi hợp lệ
        valid_move = self.board.get_valid_moves(self.current_player)
        self.valid_moves = valid_move  # Lưu trữ các nước đi hợp lệ

        for row, col in valid_move:
            cx = col * self.cell_size + self.cell_size // 2
            cy = row * self.cell_size + self.cell_size // 2
            r = 5  # bán kính dấu chấm
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="red", outline="red")
        
        # Kiểm tra nếu trò chơi kết thúc
        if self.board.is_terminal():
            self.show_result()
            return  # Dừng gọi thêm update_UI sau khi trò chơi kết thúc
        
        if not self.board.get_valid_moves(self.current_player):
            messagebox.showinfo("Mất lượt", "Không có nước đi hợp lệ! Mất lượt.")
            self.current_player = -self.current_player

            # Kiểm tra lại người chơi mới có nước đi không
            if not self.board.get_valid_moves(self.current_player):
                self.show_result()  # Cả hai đều không có nước đi → kết thúc
            else:
                self.update_UI()  # Chỉ cập nhật nếu người kia có thể đi

        
    def player_click(self, event):
        row = event.y // self.cell_size
        col = event.x // self.cell_size

        if not self.board.is_valid_move((row, col), self.current_player):
            return

        self.board.make_move((row, col), self.current_player)
        self.current_player = -self.current_player
        self.update_UI()
        
        if self.mode == "PvAI":
            self.player_vs_agent(self.player, self.depth)
    
    def agent_click(self, depth):
        step = [0]
        process = psutil.Process(os.getpid())
        start_time = time.time()
        cpu_start = process.cpu_times()
        mem_start = process.memory_info().rss
        
        
        
        
        score, move = minimax(self.board, depth, self.current_player, True, step = step)
        
        
        cpu_end = process.cpu_times()
        mem_end = process.memory_info().rss
        elapsed = time.time() - start_time
        cpu_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        mem_used = (mem_end - mem_start) / 1024  # KB

        print(f"{self.current_player}, {score}, {step[0]}, {elapsed:.4f}, {cpu_used:.4f}, {mem_used:.2f}")

    
        if move:
            self.board.make_move(move, self.current_player)
            self.current_player = -self.current_player
            self.update_UI()

    def show_result(self):
        black, white = self.board.count_score()
        if black > white:
            winner = "Người chơi Đen thắng!"
        elif white > black:
            winner = "Người chơi Trắng thắng!" if self.mode == "PvP" else "Máy thắng!"
        else:
            winner = "Hòa!"
        messagebox.showinfo("Kết thúc", f"{winner}\nĐen: {black}, Trắng: {white}")
        self.root.quit()
    
    def player_vs_player(self):
        self.canvas.bind("<Button-1>", self.player_click)
        self.update_UI()


    def player_vs_agent(self, player, depth):
        if self.board.is_terminal():
            self.show_result()
            return

        if self.current_player == player:
            self.update_UI()
        else:
            self.root.after(500, lambda: self.agent_click(depth))


    def agent_vs_agent(self, depth1, depth2):
        if self.board.is_terminal():
            self.show_result()
            return

        if self.current_player == 1:
            self.agent_click(depth1)
        else:
            self.agent_click(depth2)

        # Gọi lại lượt tiếp theo sau 500ms
        self.root.after(500, lambda: self.agent_vs_agent(depth1, depth2))

            
    def test_agent(self, player, depth, seed=0):
        random.seed(seed)

        if self.board.is_terminal():
            self.show_result()
            return

        if self.current_player == player:
            self.agent_click(depth)
        else:
            valid_moves = self.board.get_valid_moves(self.current_player)
            if valid_moves:
                move = random.choice(valid_moves)
                self.board.make_move(move, self.current_player)
                self.current_player = -self.current_player
                self.update_UI()

        # Gọi lại lượt tiếp theo sau 500ms
        self.root.after(500, lambda: self.test_agent(player, depth, seed))

            

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Othello (Reversi)")
    app = GameApp(root)
    root.mainloop()
