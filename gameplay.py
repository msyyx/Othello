from othello import *
from random import randrange

#############################################
# Heuristic Based Game Playing Agent        #
#############################################

def next_state_by_heur(state, heur_fn):
    """
    Return the suggested state according to the given heuristic
    """
    successors = state.successors()
    if len(successors) == 0:
        state.side = abs(state.side - 1)
        return state
    best = [alpha_min, successors[0]]
    for s in successors:
        h_val = heur_fn(s)
        if h_val > best[0]:
            best[0] = h_val
            best[1] = s
        elif h_val == best[0]:
            if randrange(2):
                best[1] = s
    return best[1]

def next_state_manual(state, loc):
    """
    Return the state followed from a move to loc in the current state.
    """
    side_after = abs(state.side - 1)
    action = loc
    board_after = make_move(state.board, loc, state.side)
    s = othello(action, state.gval + 1, board_after, side_after, state.game_over, state)
    return s

valid_sides = {"b": black_side,
              "black": black_side,
              "B": black_side,
              "Black": black_side,
              1: black_side,
              "w": white_side,
              "white": white_side,
              "W": white_side,
              "White": white_side,
              0: white_side}


def man_vs_machine(heur_fn):
    """
    Simulate game play between a human player and computer AI
    Computer AI uses heuristic heur_fn.
    """
    # Selecting side
    side = input(" Please select your side (b/w): ")
    while side not in valid_sides:
        side = input(" Please choose either b or w: ")
    player_side = valid_sides[side]

    # Initialize game state
    state = make_init_state()
    while not goal_fn(state):
        # Print status
        print_board(state.board)
        if player_side:
            print(" Player: Black(x), Computer: White(o)")
        else:
            print(" Player: White(o), Computer: Black(x)")

        # Check whose round it is
        if player_side == state.side:
            # Get user's move
            pmoves = get_possible_moves(state.board, state.side)
            if len(pmoves) == 0:
                state = state.successors()[0]
                continue
            print(" Possible moves: {}".format(str(pmoves).strip("[]")))
            move_str = input(" Next move? (x, y): ")
            move_list = move_str.strip("()").split(",")
            if len(move_list) != 2:
                move = (-1, -1)
            else:
                move = (int(move_list[0]), int(move_list[1]))
            # Check if the given move is valid
            while not is_valid_move(state.board, move, state.side):
                move_str = input(" Please provide a valid move: ")
                move_list = move_str.strip("()").split(",")
                if len(move_list) != 2:
                    continue
                move = (int(move_list[0]), int(move_list[1]))

            state = next_state_manual(state, move)
            print(" Player played {}".format(state.action))
        else:
            state = next_state_by_heur(state, heur_fn)
            print(" Computer played {}".format(state.action))
        print()

    print(" Final Result")
    print_board(state.board)
    print_points(state.board)
    if is_winner(state.board, player_side):
        print(" Player won!!!")
    elif is_winner(state.board, abs(player_side - 1)):
        print(" Computer won!!!")
    else:
        print(" It's a Draw!")
    print(" Game Ends")

def machine_vs_machine(heur_fn1, heur_fn2):
    """
    Simulate game play between two computer AIs.
    Computer AI 1 uses heuristic heur_fn1.
    Computer AI 2 uses heuristic heur_fn2.
    """
    # Initialize game state
    state = make_init_state()
    comp1_side = randrange(2) # Randomly assign side

    while not goal_fn(state):
        # Print status
        print_board(state.board)
        if comp1_side:
            print(" Comp. AI 1: Black(x), Comp. AI 2: White(o)")
        else:
            print(" Comp. AI 1: White(o), Comp. AI 2: Black(x)")

        # Check whose round it is
        if comp1_side == state.side:
            state = next_state_by_heur(state, heur_fn1)
            print(" Comp. AI 1 played {}".format(state.action))
        else:
            state = next_state_by_heur(state, heur_fn2)
            print(" Comp. AI 2 played {}".format(state.action))
        print()

    print(" Final Result")
    print_board(state.board)
    print_points(state.board)
    if is_winner(state.board, comp1_side):
        print(" Comp. AI 1 won!!!")
    elif is_winner(state.board, abs(comp1_side - 1)):
        print(" Comp. AI 2 won!!!")
    else:
        print(" It's a Draw!")
    print(" Game Ends")



#############################################
# Game Playing Agent with GTS approach      #
#############################################

# Game Tree Search
def next_state_by_gameTreeSearch(root, leaves, heur_fn):
    is_max = True if root.state.side == root.max_side else False
    search_depth = 1
    root.resetAlphaBeta()
    parents = list()
    children = list(leaves)
    res_leaves = list()
    for depth in range(search_depth):
        parents = list(children)
        children = list()
        for node in parents:
            s_states = node.state.successors()
            for s_state in s_states:
                s_node = Game_TreeNode(s_state, node.max_side, node)
                node.children.append(s_node)
                if goal_fn(s_state):
                    res_leaves.append(s_node)
                    continue
                children.append(s_node)

    res_leaves.extend(children)
    for leaf in res_leaves:
        if is_max:
            leaf.setBeta(heur_fn(leaf.state))
        else:
            leaf.setAlpha(heur_fn(leaf.state))
        leaf.updateParent()

    best = [alpha_min, None]
    for c_node in root.children:
        if is_max:
            # if c_node.beta == root.alpha:
            #     return c_node, res_leaves
            if c_node.beta > best[0]:
                best[0] = c_node.beta
                best[1] = c_node
            elif c_node.beta == best[0]:
                if randrange(2):
                    best[1] = c_node
        else:
            # if c_node.alpha == root.beta:
            #     return c_node, res_leaves
            if c_node.alpha > best[0]:
                best[0] = c_node.alpha
                best[1] = c_node
            elif c_node.alpha == best[0]:
                if randrange(2):
                    best[1] = c_node
    return best[1], leaves

def machine_vs_GTSmachine(heur_fn1, heur_fn2):
    """
    Simulate game play between two computer AIs.
    Computer AI 1 uses heuristic heur_fn1.
    Computer AI 2 uses game tree search with evaluation function heur_fn2.
    """
    # Initialize game state
    state = make_init_state()
    comp1_side = randrange(2) # Randomly assign side
    root = None
    leaves = list()
    root, leaves = init_game_tree(state)

    while not goal_fn(state):
        # Print status
        print_board(state.board)
        if comp1_side:
            print(" Heur. AI: Black(x), GTS AI: White(o)")
        else:
            print(" Heur. AI: White(o), GTS AI: Black(x)")

        # Check whose round it is
        if comp1_side == state.side:
            state = next_state_by_heur(state, heur_fn1)
            for c_node in root.children:
                if c_node.state.action == state.action:
                    root = c_node
                    break
            print(" Heur. AI played {}".format(state.action))
        else:
            root, leaves = next_state_by_gameTreeSearch(root, leaves, heur_fn2)
            state = root.state
            print(" GTS AI played {}".format(state.action))
        print()

    print(" Final Result")
    print_board(state.board)
    print_points(state.board)
    if is_winner(state.board, comp1_side):
        print(" Heur. AI won!!!")
    elif is_winner(state.board, abs(comp1_side - 1)):
        print(" GTS AI won!!!")
    else:
        print(" It's a Draw!")
    print(" Game Ends")



if __name__ == '__main__':
    print("\n Welcome to the Game of Othello! \n")

    # Uncomment below to simulate game play between computer AI and player
    level = input(" Please input difficulty: 0 | 1 | 2 : ")
    if int(level) == 0:
        man_vs_machine(heur_rand)
    elif int(level) == 1:
        man_vs_machine(heur_combine_advan_pts)
    elif int(level) == 2:
        man_vs_machine(heur_dynamic_advan_pts)

    # Uncomment below to simulate game play between computer AIs
    # machine_vs_machine(heur_rand, heur_gametree_search)

    # Heuristic List:
    # 1. heur_rand
    # 2. heur_high_points
    # 3. heur_high_moves
    # 4. heur_high_moves_ratio
    # 5. heur_high_points_and_moves
    # 6. heur_corners_edges
    # 7. heur_high_points_and_moves_and_corners
    # 8. heur_advantage_pts
    # 9. heur_advantage_pts_with_penalty
    # 10. heur_dynamic_advan_pts
    # 11. heur_combine_advan_pts
    # 12. heur_gametree_search

    # This is still buggy
    # Uncomment below to simulate game play between
    # heuristic based and game tree search based AI.
    # machine_vs_GTSmachine(heur_rand, heur_combine_advan_pts)
