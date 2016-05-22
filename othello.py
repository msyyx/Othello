'''
othello STATESPACE
'''

from search import *
from random import randint

##################################################
# The search space class 'othello'               #
# This class is a sub-class of 'StateSpace'      #
##################################################


# Global Variables
board_size = 8
empty_tile = "."
black_disc = "x"
white_disc = "o"
black_side = 1
white_side = 0
full_points = 64


class othello(StateSpace):
    def __init__(self, action, gval, board, side, game_over, parent=None):
        """Initialize a othello search state object."""
        StateSpace.__init__(self, action, gval, parent)
        self.board = board
        self.side = side
        self.game_over = game_over


    def successors(self):
        '''Return list of othello objects that are the successors of the current object'''
        # Get a list of possible moves
        pmoves = get_possible_moves(self.board, self.side)
        successors_list = list()
        side_after = abs(self.side - 1) # Change side

        for move in pmoves:
            action = move
            board_after = make_move(self.board, move, self.side)
            s = othello(action, self.gval + 1, board_after, side_after, self.game_over, self)
            successors_list.append(s)

        # If no move available for the current side
        if len(successors_list) == 0:
            s = othello("No move", self.gval + 1, self.board, side_after, self.game_over, self)
            # If no move available for both side, the game is over
            if self.action == "No move":
                s.game_over = True
            successors_list.append(s)
        return successors_list


    def hashable_state(self):
        '''Return a data item that can be used as a dictionary key to UNIQUELY represent the state.'''
        return tuple((loc, disc) for loc, disc in self.board.items())


    def print_state(self):
        if self.parent:
            print("Action= \"{}\", S{}, g-value = {}, (From S{})".format(self.action, self.index, self.gval, self.parent.index))
        else:
            print("Action= \"{}\", S{}, g-value = {}, (Initial State)".format(self.action, self.index, self.gval))
        print_board(self.board)
        print_points(self.board)

#############################################
# Helper functions for Othello AI           #
#############################################

def goal_fn(state):
    '''Have we reached a goal state'''
    black_points, white_points = get_points(state.board)
    if black_points == 0 or white_points == 0:
        return True
    elif black_points + white_points == full_points:
        return True
    elif state.game_over:
        return True
    return False


def init_board():
    """ Initialize game board with starter configuration. """
    # A matrix storing the board configuration
    board = dict()
    for i in range(board_size):
        for j in range(board_size):
            coordinate = (i, j)
            board[coordinate] = empty_tile
    # Add starter discs
    board[(3, 3)] = black_disc
    board[(3, 4)] = white_disc
    board[(4, 3)] = white_disc
    board[(4, 4)] = black_disc
    return board


def print_board(board):
    """ Print the game board """
    print(" ----Game Board----")
    print("  0 1 2 3 4 5 6 7")
    for y in range(board_size):
        row = str(y) + "|"
        for x in range(board_size):
            coordinate = (x, y)
            if board[coordinate] == empty_tile:
                row += " |"
            else:
                row += board[coordinate] + "|"
        print(row)
    print(" ------------------")


def make_init_state():
    """ Create the initial othello state"""
    board = init_board()
    return othello("START", 0, board, black_side, False)


def print_points(board):
    """ Print the points of both sides """
    black_points, white_points = get_points(board)
    print("Black: {}, White: {}".format(black_points, white_points))


def get_points(board):
    """ Get the points on the current board. """
    black_points = 0
    white_points = 0

    # Go through board and count the points
    for loc, disc in board.items():
        if disc == black_disc:
            black_points += 1
        elif disc == white_disc:
            white_points += 1

    return black_points, white_points


def make_move(board, loc, side):
    """ Place one disc (black or white depends on side) at loc, and update the board. """
    board_after = dict(board)

    if side:
        my_side = black_disc
        against_side = white_disc
    else:
        my_side = white_disc
        against_side = black_disc

    board_after[loc] = my_side
    surroundings = get_surrounding_locs(board, loc)
    for neighbour in surroundings:
        if board[neighbour] == against_side:
            diff = (neighbour[0] - loc[0], neighbour[1] - loc[1])
            next_loc = neighbour
            can_capture = False

            while is_valid_loc(next_loc):
                if board[next_loc] == my_side:
                    can_capture = True
                    break
                if board[next_loc] == empty_tile:
                    break
                next_loc = (next_loc[0]+diff[0], next_loc[1]+diff[1])

            if not can_capture:
                continue

            next_loc = neighbour
            while is_valid_loc(next_loc):
                if board[next_loc] == my_side:
                    break
                board_after[next_loc] = my_side
                next_loc = (next_loc[0]+diff[0], next_loc[1]+diff[1])

    return board_after


def get_possible_moves(board, side):
    """ Return all the possible coordinates to move. """
    pmoves = set()

    if side:
        against_side = white_disc
    else:
        against_side = black_disc

    for loc, disc in board.items():
        if disc != against_side:
            continue
        surroundings = get_surrounding_locs(board, loc)
        for next_loc in surroundings:
            if is_valid_move(board, next_loc, side):
                pmoves.add(next_loc)

    return list(pmoves)

def is_valid_move(board, loc, side):
    """ Return True iff a move of side to loc is a valid move on board. """
    if board[loc] != empty_tile or not is_valid_loc(loc):
        return False

    if side:
        against_side = white_disc
        my_side = black_disc
    else:
        against_side = black_disc
        my_side = white_disc

    surroundings = get_surrounding_locs(board, loc)
    for neighbour in surroundings:
        if board[neighbour] != against_side:
            continue
        diff = (neighbour[0] - loc[0], neighbour[1] - loc[1])
        next_loc = neighbour
        while is_valid_loc(next_loc):
            if board[next_loc] == my_side:
                return True
            elif board[next_loc] == empty_tile:
                break
            next_loc = (next_loc[0] + diff[0], next_loc[1] + diff[1])

    return False


def is_valid_loc(loc):
    """ Return True iff loc is a valid tile on board. """
    # Not on board
    if loc[0] < 0 or loc[0] >= board_size or loc[1] < 0 or loc[1] >= board_size:
        return False
    return True


def get_surrounding_locs(board, loc):
    """ Get valid coordinates of surrounding tiles on board """
    surroundings = [[-1, -1], [-1, 0], [-1, 1],
                    [0, -1], [0, 1],
                    [1, -1], [1, 0], [1, 1]]
    valid_surroundings = list()
    for i in range(8):
        next_loc = (loc[0] + surroundings[i][0], loc[1] + surroundings[i][1])
        if is_valid_loc(next_loc):
            valid_surroundings.append(next_loc)
    return valid_surroundings

def is_winner(board, side):
    black_pts, white_pts = get_points(board)
    if side == black_side:
        return black_pts > white_pts
    else:
        return white_pts > black_pts

def get_num_possible_moves(board, side):
    return len(get_possible_moves(board, side))

def get_corner_edge_points(board):
    black = 0
    white = 0
    for i in range(8):
        if board[(i, 0)] == black_disc:
            black += 0.6
        if board[(i, 0)] == white_disc:
            white += 0.6
        if board[(i, 7)] == black_disc:
            black += 0.6
        if board[(i, 7)] == white_disc:
            white += 0.6
    for j in range(8):
        if board[(0, j)] == black_disc:
            black += 0.6
        if board[(0, j)] == white_disc:
            white += 0.6
        if board[(7, j)] == black_disc:
            black += 0.6
        if board[(7, j)] == white_disc:
            white += 0.6
    return (black, white)

def heur_rand(state):
    """
    Randomized heuristic that returns random evaluation value.
    """
    return randint(0,3)

def heur_high_points(state):
    """
    High Points Heuristic leads the search to choose the state that would
    gain the highest points for the current side.
    """
    parent_s = state.parent
    if parent_s.side:
        my_prev_points, opp_prev_points = get_points(parent_s.board)
    else:
        opp_prev_points, my_prev_points = get_points(parent_s.board)

    if state.side:
        opp_points, my_points = get_points(state.board)
    else:
        my_points, opp_points = get_points(state.board)

    return my_points - my_prev_points

def heur_high_moves(state):
    """
    High Moves Heuristic leads the search to choose the state that would
    gain the highest number of possible moves for the current side.
    """
    parent_state = state.parent
    my_prev_moves = get_num_possible_moves(parent_state.board, parent_state.side)
    my_cur_moves = get_num_possible_moves(state.board, state.side)

    return my_cur_moves - my_prev_moves

def heur_high_moves_ratio(state):
    """
    """
    min_state_side = abs(state.side - 1)
    min_state = othello(state.action, state.gval, state.board, min_state_side, state.game_over)
    max_player_moves = get_num_possible_moves(state.board, state.side)
    min_player_moves = get_num_possible_moves(min_state.board, min_state.side)

    if (max_player_moves + min_player_moves != 0):
        return (max_player_moves - min_player_moves) / (max_player_moves + min_player_moves)
    else:
        return -1

def heur_high_points_and_moves(state):
    """
    """
    points_heur_val = heur_high_points(state)
    moves_heur_val = heur_high_moves(state)

    return points_heur_val + moves_heur_val

def heur_corners_edges(state):
    """
    Value corners and edges as higher than other positions
    """
    parent_state = state.parent
    side = state.side
    parent_black, parent_white = get_corner_edge_points(parent_state.board)
    black, white = get_corner_edge_points(state.board)
    if side == black_side:
        return black - parent_black
    else:
        return white - parent_white

def heur_high_points_and_moves_and_corners(state):
    return heur_high_points_and_moves(state) + heur_corners_edges(state)


#############################################
# Heuristics that evaluate a state based on #
# its positional advantage on the board     #
#############################################

# A board that has the same dimension as the game board
# This board is intended to store the evaluation value
# for each position on the game board.
# Hence, this board would reflect the positional advantage
# on the board of the game Othello
heur_pts_board = dict()

# Point system
priority_pts = 30
highly_desirable_pts = 25
more_desirable_pts = 15
desirable_pts = 10
neutral_pts = 5
less_desirable_pts = -5
not_desirable_pts = -20

# Compute heuristic points board
for i in range(board_size):
    for j in range(board_size):
        coordinate = (i, j)
        # Corner cases
        if i == 0 and (j == 0 or j == board_size-1):
            heur_pts_board[coordinate] = highly_desirable_pts
        elif i == board_size-1 and (j == 0 or j == board_size-1):
            heur_pts_board[coordinate] = highly_desirable_pts
        # Corners in the inner edge
        elif i == j and (i == 1 or i == board_size - 1):
            heur_pts_board[coordinate] = not_desirable_pts
        # Edges
        elif i == 0 or i == board_size - 1 or j == 0 or j == board_size - 1:
            heur_pts_board[coordinate] = desirable_pts
        # Inner edge
        elif i == 1 or i == board_size - 2 or j == 1 or j == board_size - 2:
            heur_pts_board[coordinate] = less_desirable_pts
        # Centre
        else:
            heur_pts_board[coordinate] = neutral_pts

def heur_advantage_pts(state):
    """
    Returns the evaluation value that reflects the
    positional advantage on the game board of Othello.
    """
    sum_pts = 0
    if state.side:
        parent_disc = white_disc
    else:
        parent_disc = black_disc
    for i in range(board_size):
        for j in range(board_size):
            coordinate = (i, j)
            if state.board[coordinate] == parent_disc:
                sum_pts += heur_pts_board[coordinate]
    return sum_pts


def heur_advantage_pts_with_penalty(state):
    """
    Heuristic that is based on heuristic heur_advantage_pts with additional
    penalty when the opponent gets on crucial positions.
    """
    penalty_mult = 2
    sum_pts = 0
    if state.side:
        parent_disc = white_disc
    else:
        parent_disc = black_disc
    for i in range(board_size):
        for j in range(board_size):
            coordinate = (i, j)
            if state.board[coordinate] == parent_disc:
                sum_pts += heur_pts_board[coordinate]
            elif state.board[coordinate] != empty_tile:
                # Gives penalty if the opponent makes it to the edge
                if heur_pts_board[coordinate] >= desirable_pts:
                    sum_pts -= heur_pts_board[coordinate] * penalty_mult
    return sum_pts


def compute_dynamic_pts_board(state):
    """
    Compute a point-counting board that is based on heur_pts_board.
    This board is different from heur_pts_board such that it's evaluation
    also considers the real-time changes on the board.
    Hence, this board could better reflect the real positional advantage
    on the game board.
    """
    heur_dynamic_pts_board = dict(heur_pts_board)
    board = state.board
    my_disc = black_disc if state.side == black_side else white_disc
    opp_disc = white_disc if state.side == black_side else black_disc

    # Update the evaluation value on the edges of the board.
    for i in range(1, board_size-1):
        c1 = (i, 0)
        if board[c1] == my_disc:
            heur_dynamic_pts_board[c1] = desirable_pts
            if (i-1, 0) in board and board[(i-1, 0)] == empty_tile:
                heur_dynamic_pts_board[(i-1, 0)] = more_desirable_pts
            if (i+1, 0) in board and board[(i+1, 0)] == empty_tile:
                heur_dynamic_pts_board[(i+1, 0)] = more_desirable_pts
            if (i-2, 0) in board and board[(i-2, 0)] == empty_tile:
                heur_dynamic_pts_board[(i-2, 0)] = neutral_pts
            if (i+2, 0) in board and board[(i+2, 0)] == empty_tile:
                heur_dynamic_pts_board[(i+2, 0)] = neutral_pts
        elif board[c1] == opp_disc:
            heur_dynamic_pts_board[c1] = desirable_pts
            if (i-1, 0) in board and board[(i-1, 0)] == empty_tile:
                heur_dynamic_pts_board[(i-1, 0)] = less_desirable_pts
            if (i+1, 0) in board and board[(i+1, 0)] == empty_tile:
                heur_dynamic_pts_board[(i+1, 0)] = less_desirable_pts

        c2 = (i, board_size - 1)
        if board[c2] == my_disc:
            heur_dynamic_pts_board[c2] = desirable_pts
            if (i-1, board_size-1) in board and board[(i-1, board_size-1)] == empty_tile:
                heur_dynamic_pts_board[(i-1, board_size-1)] = more_desirable_pts
            if (i+1, board_size-1) in board and board[(i+1, board_size-1)] == empty_tile:
                heur_dynamic_pts_board[(i+1, board_size-1)] = more_desirable_pts
            if (i-2, board_size-1) in board and board[(i-2, board_size-1)] == empty_tile:
                heur_dynamic_pts_board[(i-2, board_size-1)] = neutral_pts
            if (i+2, board_size-1) in board and board[(i+2, board_size-1)] == empty_tile:
                heur_dynamic_pts_board[(i+2, board_size-1)] = neutral_pts
        elif board[(i, 0)] == opp_disc:
            heur_dynamic_pts_board[(i, 0)] = desirable_pts
            if (i-1, board_size-1) in board and board[(i-1, board_size-1)] == empty_tile:
                heur_dynamic_pts_board[(i-1, board_size-1)] = less_desirable_pts
            if (i+1, board_size-1) in board and board[(i+1, board_size-1)] == empty_tile:
                heur_dynamic_pts_board[(i+1, board_size-1)] = less_desirable_pts

        c3 = (0, i)
        if board[c3] == my_disc:
            heur_dynamic_pts_board[c3] = desirable_pts
            if (0, i-1) in board and board[(0, i-1)] == empty_tile:
                heur_dynamic_pts_board[(0, i-1)] = more_desirable_pts
            if (0, i+1) in board and board[(0, i+1)] == empty_tile:
                heur_dynamic_pts_board[(0, i+1)] = more_desirable_pts
            if (0, i-2) in board and board[(0, i-2)] == empty_tile:
                heur_dynamic_pts_board[(0, i-2)] = neutral_pts
            if (0, i+2) in board and board[(0, i+2)] == empty_tile:
                heur_dynamic_pts_board[(0, i+2)] = neutral_pts
        elif board[c3] == opp_disc:
            heur_dynamic_pts_board[c3] = desirable_pts
            if (0, i-1) in board and board[(0, i-1)] == empty_tile:
                heur_dynamic_pts_board[(0, i-1)] = less_desirable_pts
            if (0, i+1) in board and board[(0, i+1)] == empty_tile:
                heur_dynamic_pts_board[(0, i+1)] = less_desirable_pts

        c4 = (board_size-1, i)
        if board[c4] == my_disc:
            heur_dynamic_pts_board[c4] = desirable_pts
            if (board_size-1, i-1) in board and board[(board_size-1, i-1)] == empty_tile:
                heur_dynamic_pts_board[(board_size-1, i-1)] = more_desirable_pts
            if (board_size-1, i+1) in board and board[(board_size-1, i+1)] == empty_tile:
                heur_dynamic_pts_board[(board_size-1, i+1)] = more_desirable_pts
            if (board_size-1, i-2) in board and board[(board_size-1, i-2)] == empty_tile:
                heur_dynamic_pts_board[(board_size-1, i-2)] = neutral_pts
            if (board_size-1, i+2) in board and board[(board_size-1, i+2)] == empty_tile:
                heur_dynamic_pts_board[(board_size-1, i+2)] = neutral_pts
        elif board[c4] == opp_disc:
            heur_dynamic_pts_board[c4] = desirable_pts
            if (board_size-1, i-1) in board and board[(board_size-1, i-1)] == empty_tile:
                heur_dynamic_pts_board[(board_size-1, i-1)] = less_desirable_pts
            if (board_size-1, i+1) in board and board[(board_size-1, i+1)] == empty_tile:
                heur_dynamic_pts_board[(board_size-1, i+1)] = less_desirable_pts

    return heur_dynamic_pts_board

def heur_dynamic_advan_pts(state):
    """
    Returns the evaluation value that reflects the positional advantage
    on the current board with dynamically updated evaluation board.
    """
    sum_pts = 0
    heur_dynamic_pts_board = compute_dynamic_pts_board(state)
    if state.side:
        parent_disc = white_disc
    else:
        parent_disc = black_disc
    for i in range(board_size):
        for j in range(board_size):
            coordinate = (i, j)
            if state.board[coordinate] == parent_disc:
                sum_pts += heur_dynamic_pts_board[coordinate]
    return sum_pts


def heur_combine_advan_pts(state):
    """
    Heuristic that combines the theoretical positional advantage evaluation
    from heuristic heur_advantage_pts and the real-time on-board advantage
    evaluation from heuristic heur_high_points.
    """
    return heur_advantage_pts(state) + heur_high_points(state)


#############################################
# Game Tree Search for Othello              #
#############################################

alpha_min = -10 * 64
beta_max = 10 * 64

class Game_TreeNode():
    """
    A data structure that represents a node on a game tree of Othello
    """
    def __init__(self, state, max_side, parent=None):
        """ Constructor """
        self.state = state
        self.max_side = max_side
        self.alpha = alpha_min
        self.beta = beta_max
        self.children = list()
        self.parent = parent

    def setAlpha(self, alpha):
        """ Set alpha value. """
        self.alpha = alpha

    def setBeta(self, beta):
        """ Set beta value. """
        self.beta = beta

    def updateParent(self):
        """ Update parent's alpha/beta value. """
        # If this is root
        if self.parent == None:
            return
        if self.state.side == self.max_side:
            # This is a max node, and its parent is a min node
            if self.alpha < self.parent.beta:
                self.parent.setBeta(self.alpha)
                self.parent.updateParent()
        else:
            # This is a min node, and its parent is a max node
            if self.beta > self.parent.alpha:
                self.parent.setAlpha(self.beta)
                self.parent.updateParent()

    def resetAlphaBeta(self):
        """ Reset the alpha/beta value of all nodes in the successors. """
        self.setAlpha(alpha_min)
        self.setBeta(beta_max)
        for child in self.children:
            child.resetAlphaBeta()

def heur_gametree_search(state):
    """
    Heuristic that looks ahead to evaluate the future board configurations
    followed from current state.
    This heuristic is based on the minimax strategy which assumes that
    both players play rationaly and only choose the best move that gives
    them the greatest advantage.
    """
    search_depth = 2 # Looks 2 plays ahead
    max_side = state.side
    root = Game_TreeNode(state, max_side)
    parents = list()
    children = [root]
    leaves = list()
    for depth in range(search_depth):
        parents = list(children)
        children = list()
        for node in parents:
            s_states = node.state.successors()
            for s_state in s_states:
                s_node = Game_TreeNode(s_state, max_side, node)
                if goal_fn(s_state):
                    leaves.append(s_node)
                    node.children.append(s_node)
                    continue
                children.append(s_node)
                node.children.append(s_node)

    leaves.extend(children)
    for leaf in leaves:
        leaf.alpha = heur_combine_advan_pts(leaf.state)
        leaf.updateParent()

    return root.alpha

def init_game_tree(init_state):
    """
    Initialize an Othello game tree with the given init_state as root.
    """
    search_depth = 4
    max_side = init_state.side
    root = Game_TreeNode(init_state, max_side)
    parents = list()
    children = [root]
    leaves = list()
    for depth in range(search_depth):
        parents = list(children)
        children = list()
        for node in parents:
            s_states = node.state.successors()
            for s_state in s_states:
                s_node = Game_TreeNode(s_state, max_side, node)
                node.children.append(s_node)
                if goal_fn(s_state):
                    leaves.append(s_node)
                    continue
                children.append(s_node)

    leaves.extend(children)
    for leaf in leaves:
        leaf.setAlpha(heur_combine_advan_pts(leaf.state))
        leaf.updateParent()

    return root, leaves

#############################################
# Main                                      #
#############################################

# if __name__ == '__main__':
   # board = init_board()
   # print_board(board)
   # board = make_move(board, (2, 4), 1)
   # board = make_move(board, (2, 5), 0)
   # board = make_move(board, (3, 5), 1)
   # print_board(board)
   # for move in get_possible_moves(board, 0):
   #     print(move)
   # s0 = make_init_state()
   # se = SearchEngine('best_first', 'full')
   # final = se.search(s0, goal_fn, heur_rand)
   # heur_gametree_search(s0)
