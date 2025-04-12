import copy
from time import sleep as delay

# global variables implementation
GREEN_COLOR_CODE = "\033[92m"
YELLOW_COLOR_CODE = "\033[93m"
RESET_COLOR_CODE = "\033[0m"
SEARCH_DEPTH = 4
PLAYER_X = "X"
PLAYER_O = "O"
BOARD_SIZE = 8
EMPTY_CELL = "."
BOARD = [[EMPTY_CELL for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]
BOARD[BOARD_SIZE//2-1][BOARD_SIZE//2-1] , BOARD[BOARD_SIZE//2-1][BOARD_SIZE//2] = PLAYER_X , PLAYER_O
BOARD[BOARD_SIZE//2][BOARD_SIZE//2-1] , BOARD[BOARD_SIZE//2][BOARD_SIZE//2] = PLAYER_O , PLAYER_X
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),(0, -1),(0, 1),(1, -1), (1, 0), (1, 1)]

# print board 
def print_board(board) : 
    print(" " , end=" ")
    for i in range(BOARD_SIZE) :
        print(f"{YELLOW_COLOR_CODE}{chr(i + 65)}{RESET_COLOR_CODE}", end=" ")
    
    print()

    for i in range(BOARD_SIZE) :
        print(f"{YELLOW_COLOR_CODE}{i + 1}{RESET_COLOR_CODE}", end=" ")
        for j in range(BOARD_SIZE) :
            print(board[i][j] , end=" ")
        print()

# checking whether if a move is valid or not 
def is_valid_move(board , row , col , player) :
    if board[row][col] != EMPTY_CELL and not is_game_over(board) :
        return False

    OPPONENT = PLAYER_O if player == PLAYER_X else PLAYER_X

    for dr , dc in DIRECTIONS :
        r, c = row + dr , col + dc 
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == OPPONENT and board[r][c] != EMPTY_CELL:
            r += dr 
            c += dc 
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player :
                return True
    return False

# finding all valid moves based on is_valid_move function
def all_valid_moves(board , player) :
    valid_moves = []

    for i in range(BOARD_SIZE) :
        for j in range(BOARD_SIZE) :
            if is_valid_move(board , i , j , player) :
                valid_moves.append((i , j))
    return valid_moves

# doing movements based on validation , flip opponent's bead if it's in a line  
def make_move(board , row , col , player) :
    if not is_valid_move(board , row , col , player) :
        return False
    
    OPPONENT = PLAYER_O if player == PLAYER_X else PLAYER_X
    board[row][col] = player

    for dr , dc in DIRECTIONS :
        r , c = row + dr , col + dc 
        to_flip = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == OPPONENT and board[r][c] != EMPTY_CELL:
            to_flip.append((r , c))
            r += dr
            c += dc
            if 0<= r < BOARD_SIZE and 0<= c < BOARD_SIZE and board[r][c] == player :
                for flip_r , flip_c in to_flip :
                    board[flip_r][flip_c] = player
    return True 

# counting players score (how many X and O are placed on the board)
def count_score(board) :
    x_count = sum(row.count(PLAYER_X) for row in board)
    o_count = sum(row.count(PLAYER_O) for row in board)
    return x_count , o_count

# if there's no empty cell then game is over 
def is_game_over(board) :
    return sum(row.count(EMPTY_CELL) for row in board) == 0

# decision making based on minimax algorithm (used for computer movements)
def minimax(board , maximizing_player , depth , alpha , beta , player) :
    if depth == 0 or is_game_over(board) :
        return count_score(board)[0] if player == PLAYER_X else count_score(board)[1], None, None
    
    OPPONENT = PLAYER_O if player == PLAYER_X else PLAYER_X
    

    if all_valid_moves(board , player) :
        valid_moves = all_valid_moves(board , player)
    else : 
        return 0 , 0 , 0 

    if maximizing_player :
        best_row, best_col = None, None
        max_eval = float("-inf")

        for move in valid_moves :
            new_board = copy.deepcopy(board)
            make_move(new_board , move[0] , move[1] , player)
            eval , _ , _ = minimax(new_board , False , depth - 1 , alpha , beta , OPPONENT) 

            if eval > max_eval : # gains the most valuable move to attack opponent (mostly computer)
                max_eval = eval
                best_row, best_col = move

            alpha = max(alpha , eval) 

            if beta <= alpha :
                break

        return max_eval, best_row, best_col
    
    else : 
        min_eval = float("inf")
        best_row, best_col = None, None

        for move in valid_moves :
            new_board = copy.deepcopy(board)
            make_move(new_board , move[0] , move[1] , player)
            eval , _ , _ = minimax(new_board , True , depth - 1 , alpha , beta , OPPONENT)
            
            if eval < min_eval : # make the opponent get the lowest score possible
                min_eval = eval
                best_row, best_col = move
        
            beta = min(beta , eval)

            if beta >= alpha : 
                break
        
        return min_eval, best_row, best_col

# print available movements using (all_valid_moves)
def print_board_with_avail_movements(board , player) :
    valid_moves = all_valid_moves(board , player)
    new_board = copy.deepcopy(board)
    for move_r, move_c in valid_moves :
            new_board[move_r][move_c] = f"{GREEN_COLOR_CODE}*{RESET_COLOR_CODE}"
    print_board(new_board)

# implementation of the way that human player can make movements and play
def human_player_logic(board , current_player):
    while True:
        try:
            print_board_with_avail_movements(board , current_player)
            row = int(input(f"player {current_player} Enter row number (1-8): ")) - 1
            col = ord(input(f"player {current_player} Enter column letter (A-H): ").upper()) - 65

            if is_valid_move(board, row, col, current_player):
                make_move(board, row, col, current_player)
                break

            else:
                print("Invalid move! Try again.")
        except ValueError:
            print("Invalid input! Please enter valid row and column.")

# it's obvious :)
def play_game_with_ai():
    board = copy.deepcopy(BOARD)
    current_player = PLAYER_X
    depth = SEARCH_DEPTH  

    while not is_game_over(board):
        # sys("clear")
        if current_player == PLAYER_X:
            human_player_logic(board , current_player)

        else: 
            _, best_row, best_col = minimax(board, True , depth , float('-inf'), float('inf'), current_player)
            make_move(board, best_row, best_col, current_player)
            print(f"Computer plays: {chr(65 + best_col)}{best_row + 1}")

        current_player = PLAYER_X if current_player == PLAYER_O else PLAYER_O
    
    x_score , o_score = count_score(board)
    print_board(board)
    print(f"X Score: {x_score}, O Score: {o_score}")

# game option 
def play_game_with_player2():
    board = copy.deepcopy(BOARD)
    current_player = PLAYER_X

    while not is_game_over(board):
        # sys("clear")
        human_player_logic(board , current_player)
        current_player = PLAYER_X if current_player == PLAYER_O else PLAYER_O

    print_board(board)
    x_score, o_score = count_score(board)
    print(f"X Score: {x_score}, O Score: {o_score}")

# game option 
def computer_vs_computer() :
    board = copy.deepcopy(BOARD)
    current_player= PLAYER_X
    depth = SEARCH_DEPTH 
    
    while not is_game_over(board) :
        if current_player == PLAYER_X :
            _ , best_row , best_col = minimax(board , True , depth , float("-inf") , float("+inf") , current_player)
            make_move(board , best_row , best_col , current_player)
        else : 
            _ , best_row , best_col = minimax(board , False , depth , float("-inf") , float("+inf") , current_player)
            make_move(board , best_row , best_col , current_player)
        
        current_player = PLAYER_X if current_player == PLAYER_O else PLAYER_O 
    
        delay(1)
        print_board(board)
        print(f"Computer {current_player} plays: {chr(65 + best_col)}{best_row + 1}")
        
    x_score , o_score = count_score(board)
    print(f"X score : {x_score} , O score {o_score}")

import tkinter as tk

# Thêm các hàm và biến cần thiết từ mã trò chơi của bạn vào đây...

class ReversiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reversi")

        self.board = copy.deepcopy(BOARD)
        self.current_player = PLAYER_X
        self.depth = SEARCH_DEPTH

        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        cell_size = 50

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                x0, y0 = j * cell_size, i * cell_size
                x1, y1 = x0 + cell_size, y0 + cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="green", outline="black")
                if self.board[i][j] != EMPTY_CELL:
                    color = "black" if self.board[i][j] == PLAYER_X else "white"
                    self.canvas.create_oval(x0+5, y0+5, x1-5, y1-5, fill=color)

    def check_game_over(self):
        if is_game_over(self.board):
            x_score, o_score = count_score(self.board)
            winner = "X" if x_score > o_score else "O" if o_score > x_score else "Draw"
            print(f"Game Over! Winner: {winner}")

    def show_available_moves(self):
     valid_moves = all_valid_moves(self.board, self.current_player)
     for move_row, move_col in valid_moves:
        x0, y0 = move_col * 50, move_row * 50
        x1, y1 = x0 + 50, y0 + 50
        self.canvas.create_oval(x0+20, y0+20, x1-20, y1-20, fill="yellow", outline="black")

    def handle_click(self, event):
     if self.current_player == PLAYER_X and not is_game_over(self.board):
        col = event.x // 50
        row = event.y // 50
        if is_valid_move(self.board, row, col, self.current_player):
            make_move(self.board, row, col, self.current_player)
            self.current_player = PLAYER_O
            self.draw_board()
            self.show_available_moves()  # Hiển thị các ô mà người chơi có thể di chuyển
            self.check_game_over()
            self.ai_move()

    def ai_move(self):
     if self.current_player == PLAYER_O and not is_game_over(self.board):
        _, best_row, best_col = minimax(self.board, True, self.depth, float('-inf'), float('inf'), self.current_player)
        make_move(self.board, best_row, best_col, self.current_player)
        self.current_player = PLAYER_X
        self.draw_board()
        self.check_game_over()
        self.canvas.after(1000, self.show_available_moves)  # Hiển thị các ô mà người chơi có thể di chuyển
        self.canvas.after(1500, self.ai_move)



def main():
    root = tk.Tk()
    game_gui = ReversiGUI(root)
    root.bind("<Button-1>", game_gui.handle_click)
    root.mainloop()

if __name__ == "__main__":
    main()
