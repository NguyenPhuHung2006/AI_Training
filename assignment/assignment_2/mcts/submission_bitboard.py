import numpy as np
import time
import random

class Node:
    def __init__(self):
        self.n_visits = 0
        self.n_wins = 0
        self.children = {}
        
    def compute_uct(self, c, n_prev_visits):
        if self.n_visits == 0:
            return np.inf
        return (self.n_wins / self.n_visits) + c * np.sqrt(np.log(n_prev_visits) / self.n_visits)
        
    def get_max_uct(self, c):
        node = None
        max_uct = -np.inf
        max_col = None
        for col, child in enumerate(self.children):
            if child is None:
                continue
            uct = child.compute_uct(c, self.n_visits)
            if uct > max_uct:
                node = child
                max_uct = uct
                max_col = col
        
        return node, max_col
    
    def get_max_visit(self):
        max_visit = -np.inf
        max_cols = None
        for cols, child in enumerate(self.children):
            if child is None:
                continue
            if child.n_visits > max_visit:
                max_cols = cols
                max_visit = child.n_visits
        return max_cols  

def convert_bitboard(board, my_mark, n_rows, n_cols):
    board_mask = 0
    player = 0
    
    for col in range(n_cols):
        for row in range(n_rows):
            bit_index = row + col * (n_rows + 1)
            cell = board[row * n_cols + col]
            
            if cell != 0:
                board_mask |= 1 << bit_index
            if cell == my_mark:
                player |= 1 << bit_index
                
    return board_mask, player

def is_win(player, n_rows, inarow):
    dirs = (1, n_rows + 1, n_rows, n_rows + 2)
        
    for d in dirs:
        m = player
        
        for _ in range(inarow - 1):
            m = m & (m >> d)
            if m == 0:
                break
            
        if m != 0:
            return True
        
    return False

def can_play(board_mask, col, top_mask):
    return board_mask & top_mask[col] == 0

def play_move(board_mask, player, col, bottom_mask):
    move = board_mask + bottom_mask[col]
    board_mask |= move
    player ^= move
    return board_mask, player

def get_opponent_mask(board_mask, player):
    return board_mask ^ player
    
def avoid_losing_moves(board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow):
    safe_moves = []
    opponent = get_opponent_mask(board_mask, player)
    
    for col in range(n_cols):
        if not can_play(board_mask, col, top_mask):
            continue
        
        next_board_mask, _ = play_move(board_mask, player, col, bottom_mask)
        
        # check if the opponent wins next
        losing = False
        for opponent_col in range(n_cols):
            if not can_play(next_board_mask, opponent_col, top_mask):
                continue
            
            _, next_opponent = play_move(next_board_mask, opponent, opponent_col, bottom_mask)
            
            if is_win(next_opponent, n_rows, inarow):
                losing = True
                break
        
        if not losing:
            safe_moves.append(col)
            
    return safe_moves

def roll_out(board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow):
    
    current = player
    B = board_mask
    is_winning = True
    
    while True:
        valid = [c for c in range(n_cols) if can_play(B, c, top_mask)]
        
        if not valid:
            return 0

        col = random.choice(valid)
        B, current = play_move(B, current, col, bottom_mask)
        
        if is_win(current, n_rows, inarow):
            return 1 if is_winning else -1
        
        current = get_opponent_mask(B, current)
        is_winning ^= 1
        
def dfs(node: Node, board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow, table):
    
    if node.n_visits == 0:
        node.n_visits = 1
        result = roll_out(board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow)
        node.n_wins += result
        return result
    
    node.n_visits += 1
    opponent = get_opponent_mask(board_mask, player)
    valid_moves = None
            
    
            

    
    
root = None
prev_board = None
table = None
def act(observation, configuration):
    n_rows = configuration.rows
    n_cols = configuration.columns
    inarow = configuration.inarow
    board = observation.board
    my_mark = observation.mark
            
    board_mask, player = convert_bitboard(board, my_mark)
    
    top_mask = ((1 << c * (n_rows + 1)) for c in range(n_cols))
    bottom_mask = ((1 << (c * (n_rows + 1) + n_rows - 1)) for c in range(n_cols))
    
    
    
    
    