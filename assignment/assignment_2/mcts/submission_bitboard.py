import numpy as np
import time
import random

class Node:
    def __init__(self, n_cols=7):
        self.n_visits = 0
        self.n_wins = 0
        self.children = {}
        self.valid_moves = {}
        self.untried_moves = {cols for cols in range(n_cols)}
        
    def compute_uct(self, c, n_prev_visits):
        if self.n_visits == 0:
            return np.inf
        return (self.n_wins / self.n_visits) + c * np.sqrt(np.log(n_prev_visits) / self.n_visits)
        
    def get_max_uct(self, c):
        node = None
        max_uct = -np.inf
        max_col = None
        for col, child in self.children.items():
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
        for cols, child in self.children.items():
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

def center_bias(cols, n_cols):
    center = n_cols // 2
    return min(cols, key=lambda c: abs(c - center))
    
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

def immediate_win(board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow):
    for col in range(n_cols):
        if not can_play(board_mask, col, top_mask):
            continue
        
        _, next_player = play_move(board_mask, player, col, bottom_mask)
        
        if is_win(next_player, n_rows, inarow):
            return col
    return None

def find_opponent_col(board_mask, prev_board_mask, n_rows):
    move = board_mask ^ prev_board_mask
    
    if move == 0:
        return None
    
    bit_index = move.bit_length() - 1
    
    col = bit_index // n_rows
    
    return col

def roll_out(board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow):
    current = player
    B = board_mask
    is_winning = True
    
    while True:
        valid = [c for c in range(n_cols) if can_play(B, c, top_mask)]
        
        if not valid:
            return 0
        
        opponent = get_opponent_mask(B, current)
        win_move = immediate_win(B, current, top_mask, bottom_mask, n_rows, n_cols, inarow)
        block_move = immediate_win(B, opponent, top_mask, bottom_mask, n_rows, n_cols, inarow)
        col = None
        
        if block_move is not None:
            col = block_move
        if win_move is not None:
            col = win_move
        if col is None:
            col = random.choice(valid)
    
        B, current = play_move(B, current, col, bottom_mask)
        
        if is_win(current, n_rows, inarow):
            return 1 if is_winning else -1
        
        current = get_opponent_mask(B, current)
        is_winning ^= 1
        
def dfs(node: Node, c, board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow, table):
    if node.n_visits == 0:
        node.n_visits = 1
        result = roll_out(board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow)
        node.n_wins += result
        node.valid_moves = {c for c in range(n_cols) if can_play(board_mask, c, top_mask)}
        return result
    
    opponent = get_opponent_mask(board_mask, player)
    node.n_visits += 1    
    next_cols = node.valid_moves & node.untried_moves
    child = None
    next_col = None
    
    if next_cols:
        next_col = center_bias(next_cols, n_cols)
        
        next_board_mask, next_player = play_move(board_mask, player, next_col, bottom_mask)
        board_hash = (next_board_mask, next_player)
        
        if board_hash in table:
            child = table[board_hash]
        else:
            child = Node(n_cols)
            table[board_hash] = child
            
        node.untried_moves.discard(next_col)
        node.children[next_col] = child
    else:
        child, next_col = node.get_max_uct(c)
        
    # the board is full
    if child is None or next_col is None:
        if is_win(player, n_rows, inarow):
            return 1
        if is_win(opponent, n_rows, inarow):
            return -1
        return 0
    
    next_board_mask, _ = play_move(board_mask, player, next_col, bottom_mask)
    
    result = -dfs(
        child, 
        c, 
        next_board_mask,
        opponent,
        top_mask,
        bottom_mask,
        n_rows,
        n_cols,
        inarow,
        table    
    )
    
    node.n_wins += result

    return result
    
root = None
prev_board_mask = 0
table = None
def act(observation, configuration):
    n_rows = configuration.rows
    n_cols = configuration.columns
    inarow = configuration.inarow
    board = observation.board
    my_mark = observation.mark
            
    board_mask, player = convert_bitboard(board, my_mark, n_rows, n_cols)
    
    top_mask = [(1 << c * (n_rows + 1)) for c in range(n_cols)]
    bottom_mask = [(1 << (c * (n_rows + 1) + n_rows - 1)) for c in range(n_cols)]
    
    global root, prev_board_mask, table
    if not root:
        root = Node(n_cols)  
        table = {}
    if prev_board_mask > 0:
        opponent_col = find_opponent_col(board_mask, prev_board_mask, n_rows)
        if opponent_col is not None and root.children[opponent_col]:
            root = root.children[opponent_col]
        else:
            root = Node(n_cols)
            
    # board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow
    opponent = get_opponent_mask(board_mask, player)
    win_move = immediate_win(board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow)
    block_move = immediate_win(board_mask, opponent, top_mask, bottom_mask, n_rows, n_cols, inarow)
    
    immediate_move = None
    if block_move:
        immediate_move = block_move
    if win_move:
        immediate_move = win_move
            
    if immediate_move is not None:
        if root.children[immediate_move]:
            root = root.children[immediate_move]
        else:
            root = Node(n_cols)

        prev_board_mask = board_mask
        return immediate_move
    
    c = 1.36
    start = time.time()
    while time.time() - start < 1.8:
        dfs(root, c, board_mask, player, top_mask, bottom_mask, n_rows, n_cols, inarow, table)
    
    next_col = root.get_max_visit()
    root = root.children[next_col]
    prev_board_mask = board_mask
    return next_col    
        
    