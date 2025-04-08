class Board:
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    def __init__(self, size=8):
        if size % 2 != 0:
            raise ValueError("Board size must be even.")
        if size < 4:
            raise ValueError("Board size must be at least 4.")
        if size > 26:
            raise ValueError("Board size must be at most 26.")
        self.size = size
        self.board = [[self.EMPTY for _ in range(size)] for _ in range(size)]
        self.initialize()

    def initialize(self):
        mid = self.size // 2
        self.board[mid - 1][mid - 1] = self.WHITE
        self.board[mid][mid] = self.WHITE
        self.board[mid - 1][mid] = self.BLACK
        self.board[mid][mid - 1] = self.BLACK

    def __getitem__(self, index):
        return self.board[index]  # Return raw row for logic

    def __setitem__(self, index, value):
        self.board[index] = value

    def __repr__(self):
        board_str = "  " + " ".join(str(i) for i in range(self.size)) + "\n"
        for idx, row in enumerate(self.board):
            board_str += str(idx) + " "
            board_str += " ".join(self._to_str(cell) for cell in row) + "\n"
        return board_str

    def _to_str(self, value):
        if value == self.BLACK:
            return '●'
        elif value == self.WHITE:
            return '○'
        else:
            return '.'


class Othello:
    def __init__(self, size=8):
        self.board = Board(size)
        self.player = Board.BLACK

    def valid_moves(self, index, player):
        row, col = index
        if not (0 <= row < self.board.size and 0 <= col < self.board.size):
            return False
        if self.board[row][col] != Board.EMPTY:
            return False
        return True

    def make_move(self, index):
        row, col = index
        if not self.valid_moves(index, self.player):
            raise ValueError("Invalid move.")

        self.board[row][col] = self.player
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        opponent = -self.player
        for dr, dc in directions:
            r, c = row + dr, col + dc
            pieces_to_flip = []
            while 0 <= r < self.board.size and 0 <= c < self.board.size:
                if self.board[r][c] == opponent:
                    pieces_to_flip.append((r, c))
                elif self.board[r][c] == self.player:
                    for rr, cc in pieces_to_flip:
                        self.board[rr][cc] = self.player
                    break
                else:
                    break
                r += dr
                c += dc

        self.player = Board.WHITE if self.player == Board.BLACK else Board.BLACK
        return True

game = Othello(8)
print(game.board)

game.make_move((2, 2))
print(game.board)

game.make_move((5, 3))
print(game.board)

game.make_move((2, 3))
print(game.board)
game.make_move((6, 3))
print(game.board)
game.make_move((7, 3))
print(game.board)