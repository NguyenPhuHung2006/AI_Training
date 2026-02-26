import numpy as np
import time
import random

class Node:
    def __init__(self, n_cols=7):
        self.n_visits = 0
        self.n_wins = 0
        self.children = {}
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

def convert_bitboard(board):
    board_mask = 0
    player = 0
    
    for col in range(n_cols):
        for row in range(n_rows):
            bit_index = (n_rows - row - 1) + col * (n_rows + 1)
            cell = board[col + row * n_cols]
            
            if cell != 0:
                board_mask |= 1 << bit_index
            if cell == my_mark:
                player |= 1 << bit_index
                
    return board_mask, player

def is_win(player):
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

def can_play(board_mask, col):
    return board_mask & top_mask[col] == 0

def play_move(board_mask, player, col):
    move = board_mask + bottom_mask[col]
    flipped_board_mask = ~board_mask & ((1 << ((n_rows + 1) * n_cols)) - 1)
    new_piece = flipped_board_mask & move
    board_mask |= new_piece
    player |= new_piece
    return board_mask, player

def get_opponent_mask(board_mask, player):
    return board_mask ^ player

def center_bias(cols):
    center = n_cols // 2
    return min(cols, key=lambda c: abs(c - center))
    
def avoid_losing_moves(board_mask, player):
    safe_moves = []
    opponent = get_opponent_mask(board_mask, player)
    
    for col in range(n_cols):
        if not can_play(board_mask, col):
            continue
        
        next_board_mask, _ = play_move(board_mask, player, col)
        
        # check if the opponent wins next
        losing = False
        for opponent_col in range(n_cols):
            if not can_play(next_board_mask, opponent_col):
                continue
            
            _, next_opponent = play_move(next_board_mask, opponent, opponent_col)
            
            if is_win(next_opponent):
                losing = True
                break
        
        if not losing:
            safe_moves.append(col)
            
    return safe_moves

def immediate_win(board_mask, player):
    for col in range(n_cols):
        if not can_play(board_mask, col):
            continue
        
        _, next_player = play_move(board_mask, player, col)
        
        if is_win(next_player):
            return col
    return None

def find_opponent_col(board_mask, prev_board_mask):
    move = board_mask ^ prev_board_mask
    
    if move == 0:
        return None
    
    bit_index = move.bit_length() - 1
    
    col = bit_index // (n_rows + 1)
    
    return col

def roll_out(board_mask, player, is_root_turn):
    current = player
    is_winning = True
    
    while True:
        valid = [c for c in range(n_cols) if can_play(board_mask, c)]
        
        if not valid:
            return draw_score
        
        opponent = get_opponent_mask(board_mask, current)
        win_move = immediate_win(board_mask, current)
        block_move = immediate_win(board_mask, opponent)
        col = None
        
        if block_move is not None:
            col = block_move
        if win_move is not None:
            col = win_move
        if col is None:
            col = random.choice(valid)
    
        board_mask, current = play_move(board_mask, current, col)
        
        if is_win(current):
            is_root_winning = is_winning ^ is_root_turn
            return lose_score if is_root_winning else win_score
        
        current = get_opponent_mask(board_mask, current)
        is_winning ^= 1
        
def dfs(node: Node, 
        c, 
        board_mask, 
        player, 
        is_root_turn=True):
        
    if node.n_visits == 0:
        node.n_visits = 1
        result = roll_out(board_mask, player, is_root_turn)
        node.n_wins += result
        return result
    
    opponent = get_opponent_mask(board_mask, player)
    node.n_visits += 1
    valid_moves = {c for c in range(n_cols) if can_play(board_mask, c)}
    next_cols = valid_moves & node.untried_moves
    child = None
    next_col = None
    
    if next_cols:
        next_col = center_bias(next_cols)
        
        next_board_mask, next_player = play_move(board_mask, player, next_col)
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
        is_player_win = is_win(player)
        is_opponent_win = is_win(opponent)
        if is_player_win:
            return win_score if is_root_turn else lose_score
        if is_opponent_win:
            return lose_score if is_root_turn else win_score
        return draw_score
    
    next_board_mask, next_player = play_move(board_mask, player, next_col)
    next_opponent = get_opponent_mask(next_board_mask, next_player)
    
    result = dfs(
        child, 
        c, 
        next_board_mask,
        next_opponent,
        is_root_turn=is_root_turn ^ 1
    )
    
    node.n_wins += result

    return result

# for multiple games
def reset_agent():
    global root, prev_board_mask, table, is_init
    root = None
    prev_board_mask = 0
    table = None
    is_init = False
    
# global variables
root = None
prev_board_mask = 0
table = None
n_rows = 0
n_cols = 0
inarow = 0
my_mark = 0
bottom_mask = None
top_mask = None
is_init = False
win_score = 0
lose_score = 0
draw_score = 0
def act(observation, configuration):
        
    global is_init, n_rows, n_cols, inarow, my_mark, bottom_mask, top_mask, win_score, lose_score, draw_score
    if not is_init:
        is_init = True
        n_rows = configuration.rows
        n_cols = configuration.columns
        inarow = configuration.inarow
        my_mark = observation.mark
        bottom_mask = [(1 << c * (n_rows + 1)) for c in range(n_cols)]
        top_mask = [(1 << (c * (n_rows + 1) + n_rows - 1)) for c in range(n_cols)]
        win_score = 1
        lose_score = -1
        draw_score = 0
        
    board = observation.board
            
    board_mask, player = convert_bitboard(board)
    
    global root, prev_board_mask, table
    if not root:
        root = Node(n_cols)  
        table = {}
    if prev_board_mask > 0:
        opponent_col = find_opponent_col(board_mask, prev_board_mask)
        if opponent_col is not None and opponent_col in root.children:
            root = root.children[opponent_col]
        else:
            root = Node(n_cols)
            
    opponent = get_opponent_mask(board_mask, player)
    win_move = immediate_win(board_mask, player)
    block_move = immediate_win(board_mask, opponent)
    
    immediate_move = None
    if block_move is not None:
        immediate_move = block_move
    if win_move is not None:
        immediate_move = win_move

    if immediate_move is not None:
        if immediate_move in root.children:
            root = root.children[immediate_move]
        else:
            root = Node(n_cols)

        prev_board_mask = board_mask
        return immediate_move
    
    c = 1.36
    start = time.time()
    while time.time() - start < 1.6:
        dfs(root, c, board_mask, player)
    
    next_col = root.get_max_visit()
    root = root.children[next_col]
    prev_board_mask = board_mask
    return next_col    
        
    