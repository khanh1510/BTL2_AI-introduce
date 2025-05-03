import tkinter as tk
from tkinter import messagebox
import copy
import time

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),         (0, 1),
              (1, -1),  (1, 0), (1, 1)]
class Board:
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    def __init__(self, size=8):
        self.size = size
        self.board = [[self.EMPTY for _ in range(size)] for _ in range(size)]
        self.initialize()

    def initialize(self):
        mid = self.size // 2
        self.board[mid - 1][mid - 1] = self.WHITE
        self.board[mid][mid] = self.WHITE
        self.board[mid - 1][mid] = self.BLACK
        self.board[mid][mid - 1] = self.BLACK

    def get_valid_moves(self, player):
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.is_valid_move((r, c), player)]

    def is_valid_move(self, index, player):
        row, col = index
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.board[row][col] != self.EMPTY:
            return False
        opponent = -player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            has_opponent = False
            while 0 <= r < self.size and 0 <= c < self.size:
                if self.board[r][c] == opponent:
                    has_opponent = True
                elif self.board[r][c] == player:
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
        self.board[row][col] = player
        opponent = -player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            pieces_to_flip = []
            while 0 <= r < self.size and 0 <= c < self.size:
                if self.board[r][c] == opponent:
                    pieces_to_flip.append((r, c))
                elif self.board[r][c] == player:
                    for rr, cc in pieces_to_flip:
                        self.board[rr][cc] = player
                    break
                else:
                    break
                r += dr
                c += dc

        return True

    def count_score(self):
        black = sum(cell == self.BLACK for row in self.board for cell in row)
        white = sum(cell == self.WHITE for row in self.board for cell in row)
        return black, white

    def is_terminal(self):
        return not self.get_valid_moves(self.BLACK) and not self.get_valid_moves(self.WHITE)

    def copy(self):
        new_board = Board(self.size)
        new_board.board = copy.deepcopy(self.board)
        return new_board

weights = [
    [ 54, 51, 34, 30, 31, 32, 41, 42 ],
    [ 55, 50, 43, 33, 29, 28, 39, 58 ],
    [ 23, 27,  3,  4, 25,  8, 40, 59 ],
    [ 24, 22,  5,  0,  0,  6, 37, 60 ],
    [ 47, 20, 14,  0,  0,  1, 35, 38 ],
    [ 26, 21, 15,  2,  9,  7, 12, 36 ],
    [ 49, 56, 16, 11, 10, 18, 45, 53 ],
    [ 57, 46, 17, 13, 44, 52, 19, 48 ],
]


def evaluate(board, player):
    # 1. Disc count
    black, white = board.count_score()
    material_score = (black - white) * player
    return material_score
   

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
        self.mode = tk.StringVar(value="PvP")
        self.difficulty = tk.StringVar(value="Medium")
        self.depth_map = {"Easy": 6, "Medium": 8, "Hard": 10}
        self.setup_start_screen()

    def setup_start_screen(self):
        self.clear_root()
        tk.Label(self.root, text="Othello", font=("Helvetica", 28, "bold"), fg="#88c0d0", bg="#2e3440").pack(pady=20)

        tk.Label(self.root, text="Chọn chế độ chơi:", font=("Helvetica", 14), fg="white", bg="#2e3440").pack(pady=5)
        tk.OptionMenu(self.root, self.mode, "PvAI", "PvP", command=self.on_mode_change).pack()

        self.difficulty_frame = tk.Frame(self.root, bg="#2e3440")
        tk.Label(self.difficulty_frame, text="Chọn độ khó (AI):", font=("Helvetica", 14), fg="white", bg="#2e3440").pack(pady=5)
        tk.OptionMenu(self.difficulty_frame, self.difficulty, "Easy", "Medium", "Hard").pack()
        self.difficulty_frame.pack()

        tk.Button(self.root, text="Bắt đầu", font=("Helvetica", 14, "bold"),
                  bg="#5e81ac", fg="white", relief="flat", command=self.start_game).pack(pady=30)

        self.on_mode_change(self.mode.get())

    def on_mode_change(self, mode):
        if mode == "PvP":
            self.difficulty_frame.pack_forget()
        else:
            self.difficulty_frame.pack()

    def start_game(self):
        self.clear_root()
        self.game = OthelloGUI(self.root,
                               mode=self.mode.get(),
                               ai_depth=self.depth_map[self.difficulty.get()])
        self.game.draw_board()

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()


class OthelloGUI:
    def __init__(self, root, size=8, mode="PvAI", ai_depth=3):
        self.root = root
        self.board = Board(size)
        self.player = Board.BLACK
        self.mode = mode
        self.ai_depth = ai_depth
        self.cell_size = 60

        self.score_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#2e3440", fg="white")
        self.score_label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=size * self.cell_size, height=size * self.cell_size, bg="#3b4252", bd=0, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.total_step = 0
        self.total_score = 0




    def draw_board(self):
        self.canvas.delete("all")
        for r in range(self.board.size):
            for c in range(self.board.size):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#4c566a", outline="#2e3440", width=2)

                if self.board.board[r][c] == Board.BLACK:
                    self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="black", outline="")
                elif self.board.board[r][c] == Board.WHITE:
                    self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="white", outline="")

        black, white = self.board.count_score()
        self.score_label.config(text=f"Đen: {black} | Trắng: {white}")

    def handle_click(self, event):
        row = event.y // self.cell_size
        col = event.x // self.cell_size
        if not self.board.is_valid_move((row, col), self.player):
            return

        self.board.make_move((row, col), self.player)
        self.player = -self.player
        self.draw_board()

        if self.board.is_terminal():
            self.show_result()
            return

        # Nếu người chơi tiếp theo không có nước đi, mất lượt
        if not self.board.get_valid_moves(self.player):
            messagebox.showinfo("Mất lượt", "Không có nước đi hợp lệ! Mất lượt.")
            self.player = -self.player  # Trả lại lượt
            self.draw_board()

            if self.board.is_terminal():
                self.show_result()
                return

        if self.mode == "PvAI" and self.player == Board.WHITE:
            self.root.after(500, self.ai_move)


    def ai_move(self):
        if not self.board.get_valid_moves(self.player):
            messagebox.showinfo("Máy mất lượt", "Máy không có nước đi hợp lệ! Mất lượt.")
            self.player = -self.player
            self.draw_board()

            return
        step = [0]
        start_time = time.time()
        score, move = minimax(self.board, self.ai_depth, self.player, True, step=step)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(score, step[0], elapsed_time)


        if move:
            self.board.make_move(move, self.player)
            self.player = -self.player
            self.draw_board()

        if self.board.is_terminal():
            self.show_result()
            return

        # Nếu sau lượt của máy, người chơi cũng không có nước đi
        if not self.board.get_valid_moves(self.player):
            messagebox.showinfo("Bạn mất lượt", "Bạn không có nước đi hợp lệ! Mất lượt.")
            self.player = -self.player
            self.draw_board()

            if self.mode == "PvAI" and self.player == Board.WHITE:
                self.root.after(500, self.ai_move)


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


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Othello (Reversi)")
    app = GameApp(root)
    root.mainloop()
