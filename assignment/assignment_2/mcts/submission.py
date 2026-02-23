import numpy as np
import time
                
def get_valid_columns(board):
    return {i for i, val in enumerate(board[0]) if val == 0}

def drop_piece(board, action, mark, in_place=True):
    if not in_place:
        board = board.copy()
    n_rows = board.shape[0]

    for row in range(n_rows - 1, -1, -1):   
        if board[row, action] == 0:
            board[row, action] = mark
            return board, row
    return board, None

def check_win(board, row, col, mark, inarow):
    n_rows = len(board)
    n_cols = len(board[0])

    directions = [
        (0, 1),   # horizontal
        (1, 0),   # vertical
        (1, 1),   # diag ↘
        (1, -1)   # diag ↙
    ]

    for dr, dc in directions:
        count = 1  

        # forward direction
        rr, cc = row + dr, col + dc
        while 0 <= rr < n_rows and 0 <= cc < n_cols and board[rr][cc] == mark:
            count += 1
            rr += dr
            cc += dc

        # backward direction
        rr, cc = row - dr, col - dc
        while 0 <= rr < n_rows and 0 <= cc < n_cols and board[rr][cc] == mark:
            count += 1
            rr -= dr
            cc -= dc

        if count >= inarow:
            return True

    return False

def find_last_move(prev_board, current_board):
    diff = current_board - prev_board
    for col in range(diff.shape[1]):
        if np.any(diff[:, col] != 0):
            return col
    return None

class Node:
    def __init__(self, n_cols=7):
        self.n_visits = 0
        self.n_wins = 0
        self.children = [None] * n_cols
        self.untried_cols = {cols for cols in range(n_cols)}
        self.is_full = False
        
    def compute_uct(self, c, n_prev_visits):
        if self.n_visits == 0:
            return np.inf
        return (self.n_wins / self.n_visits) + c * np.sqrt(np.log(n_prev_visits) / self.n_visits)
        
    def get_max_uct(self, c):
        node = None
        max_uct = -np.inf
        max_cols = None
        for cols, child in enumerate(self.children):
            if child is None:
                continue
            uct = child.compute_uct(c, self.n_visits)
            if uct > max_uct:
                node = child
                max_uct = uct
                max_cols = cols
        
        return node, max_cols
    
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

def get_opponent_mark(mark):
    return 1 if mark == 2 else 2

def immediate_move(board, mark, inarow, valid_cols):
    for col in valid_cols:
        temp, row = drop_piece(board, col, mark, in_place=False)
        if check_win(temp, row, col, mark, inarow):
            return col
    return None

def get_delta(mark, my_mark, opponent_mark):
    if mark == my_mark:
        return 1
    if mark == opponent_mark:
        return -1
    return 0

def center_bias(cols, n_cols):
    center = n_cols // 2
    return min(cols, key=lambda c: abs(c - center))

def roll_out(board, mark, inarow, my_mark, opponent_mark):
    opp_mark = get_opponent_mark(mark)
    valid_cols = get_valid_columns(board)
    while valid_cols:
        
        cur_mark_cols = immediate_move(board, mark, inarow, valid_cols)
        if cur_mark_cols is not None:
            return get_delta(mark, my_mark, opponent_mark)
        
        opp_mark_cols = immediate_move(board, opp_mark, inarow, valid_cols)
        if opp_mark_cols is not None:
            return get_delta(opp_mark, my_mark, opponent_mark)
        
        col = center_bias(valid_cols, board.shape[1])
        _, row = drop_piece(board, col, mark)
        if check_win(board, row, col, mark, inarow):
            return get_delta(mark, my_mark, opponent_mark)
        
        valid_cols = get_valid_columns(board)
        if not valid_cols:
            return 0
        col = center_bias(valid_cols, board.shape[1])
        _, row = drop_piece(board, col, mark)
        if check_win(board, row, col, opp_mark, inarow):
            return get_delta(opp_mark, my_mark, opponent_mark)
        
        valid_cols = get_valid_columns(board)
        
    return 0

def dfs(node: Node, board, mark, c, my_mark, opponent_mark, inarow, table):
        
    if node.n_visits == 0:
        node.n_visits = 1
        delta = roll_out(board, mark, inarow, my_mark, opponent_mark)
        node.n_wins += delta
        return delta
        
    node.n_visits += 1
    child_node = None
    child_cols = None
    valid_cols = get_valid_columns(board)
    
    if not valid_cols:
        return 0
    
    child_cols = node.untried_cols & valid_cols
    if child_cols:
        child_cols = center_bias(child_cols, board.shape[1])
        
        next_board, _ = drop_piece(board, child_cols, mark, in_place=False)
        board_hash = tuple(next_board.flatten())
        
        if board_hash in table:
            child_node = table[board_hash]
        else:
            child_node = Node(board.shape[1])
            table[board_hash] = child_node
        
        node.untried_cols.discard(child_cols)
        node.children[child_cols] = child_node
        
    else:
        child_node, child_cols = node.get_max_uct(c)
        
    if child_cols is None:
        return 0
    
    drop_piece(board, child_cols, mark)
        
    delta = dfs(child_node, board, get_opponent_mark(mark), 
                c, my_mark, opponent_mark, inarow, table)
    node.n_wins += delta
    return delta

root = None
prev_board = None
table = None
def act(observation, configuration):
    n_rows = configuration.rows
    n_cols = configuration.columns
    inarow = configuration.inarow
    board = observation.board
    
    my_mark = observation.mark
    opponent_mark = 1 if my_mark == 2 else 2
    
    board = np.array(board).reshape(n_rows, n_cols)
        
    global root, prev_board, table
    if not root:
        root = Node(n_cols)  
        table = {}
    if prev_board is not None:
        last_opponent_cols = find_last_move(prev_board, board)
        if last_opponent_cols is not None and root.children[last_opponent_cols]:
            root = root.children[last_opponent_cols]
        else:
            root = Node(n_cols)
        
    valid_cols = get_valid_columns(board)  
    win_cols = immediate_move(board, my_mark, inarow, valid_cols)    
    if win_cols is not None:
        if root.children[win_cols]:
            root = root.children[win_cols]
        else:
            root = Node(n_cols)

        drop_piece(board, win_cols, my_mark)
        prev_board = board.copy()
        return win_cols
    
    block_cols = immediate_move(board, opponent_mark, inarow, valid_cols)
    if block_cols is not None:
        if root.children[block_cols]:
            root = root.children[block_cols]
        else:
            root = Node(n_cols)
            
        drop_piece(board, block_cols, my_mark)
        prev_board = board.copy()
        return block_cols

    c = 1.0
    start = time.time()
    while time.time() - start < 1.8:
        sim_board = board.copy()
        dfs(root, sim_board, my_mark, c, my_mark, opponent_mark, inarow, table)
    
    next_cols = root.get_max_visit()
    root = root.children[next_cols]
    drop_piece(board, next_cols, my_mark)
    prev_board = board.copy()
    return next_cols
    