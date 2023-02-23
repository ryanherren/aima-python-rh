"""
Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions.
"""

import sys
import getopt
from collections import deque

from utils import *
import time


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


# ______________________________________________________________________________
# Uninformed Search algorithms

def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    print()
                    print("Total nodes generated:", len(frontier))
                    return child
                frontier.append(child)
    return None


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            print("\nTotal nodes generated:", len(frontier))
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


cnt = 0


def depth_limited_search(problem, limit=50):
    """[Figure 3.17]"""

    def recursive_dls(node, problem, limit):
        global cnt
        if problem.goal_test(node.state):
            print("\nTotal nodes generated:", cnt)
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                cnt = cnt + 1
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    """[Figure 3.18]"""
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result



# ______________________________________________________________________________
# Informed (Heuristic) Search


greedy_best_first_graph_search = best_first_graph_search


# Greedy best-first search is accomplished by specifying f(n) = h(n).


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


# ______________________________________________________________________________
# A* heuristics 

# USED FOR BFS, IDS, and A* H1
class EightPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['D', 'U', 'R', 'L']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('R')
        if index_blank_square < 3:
            possible_actions.remove('D')
        if index_blank_square % 3 == 2:
            possible_actions.remove('L')
        if index_blank_square > 5:
            possible_actions.remove('U')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'D': -3, 'U': 3, 'R': -1, 'L': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is 
        h(n) = number of misplaced tiles """
        return sum(s != g for (s, g) in zip(node.state, self.goal))


# USED FOR A* H2
class EightPuzzle2(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['D', 'U', 'R', 'L']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('R')
        if index_blank_square < 3:
            possible_actions.remove('D')
        if index_blank_square % 3 == 2:
            possible_actions.remove('L')
        if index_blank_square > 5:
            possible_actions.remove('U')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'D': -3, 'U': 3, 'R': -1, 'L': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """ Return the heuristic value for a given state
        based on Manhattan distance from current location'
        of blank space to goal location of blank space"""

        def find_blank(node):
            num = 0
            for i in node.state:
                num = num + 1
                if (i == 0):
                    break;
            return num;

        num = find_blank(node)
        # Manhattan Heuristic Function (computations by hand, hardcoded)
        # 1 2 3
        # 4 5 6
        # 7 8 9
        # The grid above represents the numbers of each slot of the puzzle
        # Each number represents a slot with a tile
        # This function finds the spot of the blank, and then returns
        # its Manhattan distance from slot 9, the constant goal slot
        # of the blank in our current goal state
        if num in (8, 6):
            return 1
        elif num in (3, 5, 7):
            return 2
        elif num in (2, 4):
            return 3
        elif num == 1:
            return 4
        else:
            return 0


# USED FOR A* H3
class EightPuzzle3(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['D', 'U', 'R', 'L']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('R')
        if index_blank_square < 3:
            possible_actions.remove('D')
        if index_blank_square % 3 == 2:
            possible_actions.remove('L')
        if index_blank_square > 5:
            possible_actions.remove('U')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'D': -3, 'U': 3, 'R': -1, 'L': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """ Return a custom heuristic value for a given state."""
        """ Function is computing total Manhattan distance for all tiles."""
        """ Values are hard coded based on goal state """
        """ so if tile 1 is in the top left slot, distance from tile 1 goal is 0"""
        """ If tile 1 is in the middle slot, distance from tile 1 goal is 2 """
        state = node.state
        total_manhattan = 0
        for i in state:
            if i == 0:
                if state[i] == 1:
                    total_manhattan += 1
                if state[i] == 2:
                    total_manhattan += 0
                if state[i] == 3:
                    total_manhattan += 2
                if state[i] == 4:
                    total_manhattan += 1
                if state[i] == 5:
                    total_manhattan += 2
                if state[i] == 6:
                    total_manhattan += 3
                if state[i] == 7:
                    total_manhattan += 2
                if state[i] == 8:
                    total_manhattan += 3
                if state[i] == 0:
                    total_manhattan += 4
            if i == 1:
                if state[i] == 1:
                    total_manhattan += 0
                if state[i] == 2:
                    total_manhattan += 1
                if state[i] == 3:
                    total_manhattan += 1
                if state[i] == 4:
                    total_manhattan += 2
                if state[i] == 5:
                    total_manhattan += 1
                if state[i] == 6:
                    total_manhattan += 2
                if state[i] == 7:
                    total_manhattan += 3
                if state[i] == 8:
                    total_manhattan += 2
                if state[i] == 0:
                    total_manhattan += 3
            if i == 2:
                if state[i] == 1:
                    total_manhattan += 2
                if state[i] == 2:
                    total_manhattan += 1
                if state[i] == 3:
                    total_manhattan += 0
                if state[i] == 4:
                    total_manhattan += 3
                if state[i] == 5:
                    total_manhattan += 2
                if state[i] == 6:
                    total_manhattan += 1
                if state[i] == 7:
                    total_manhattan += 4
                if state[i] == 8:
                    total_manhattan += 3
                if state[i] == 0:
                    total_manhattan += 2
            if i == 3:
                if state[i] == 1:
                    total_manhattan += 1
                if state[i] == 2:
                    total_manhattan += 2
                if state[i] == 3:
                    total_manhattan += 3
                if state[i] == 4:
                    total_manhattan += 0
                if state[i] == 5:
                    total_manhattan += 1
                if state[i] == 6:
                    total_manhattan += 2
                if state[i] == 7:
                    total_manhattan += 1
                if state[i] == 8:
                    total_manhattan += 2
                if state[i] == 0:
                    total_manhattan += 3
            if i == 4:
                if state[i] == 1:
                    total_manhattan += 2
                if state[i] == 2:
                    total_manhattan += 1
                if state[i] == 3:
                    total_manhattan += 2
                if state[i] == 4:
                    total_manhattan += 1
                if state[i] == 5:
                    total_manhattan += 0
                if state[i] == 6:
                    total_manhattan += 1
                if state[i] == 7:
                    total_manhattan += 2
                if state[i] == 8:
                    total_manhattan += 1
                if state[i] == 0:
                    total_manhattan += 2
            if i == 5:
                if state[i] == 1:
                    total_manhattan += 3
                if state[i] == 2:
                    total_manhattan += 2
                if state[i] == 3:
                    total_manhattan += 1
                if state[i] == 4:
                    total_manhattan += 2
                if state[i] == 5:
                    total_manhattan += 1
                if state[i] == 6:
                    total_manhattan += 0
                if state[i] == 7:
                    total_manhattan += 3
                if state[i] == 8:
                    total_manhattan += 2
                if state[i] == 0:
                    total_manhattan += 1
            if i == 6:
                if state[i] == 1:
                    total_manhattan += 2
                if state[i] == 2:
                    total_manhattan += 3
                if state[i] == 3:
                    total_manhattan += 4
                if state[i] == 4:
                    total_manhattan += 1
                if state[i] == 5:
                    total_manhattan += 2
                if state[i] == 6:
                    total_manhattan += 3
                if state[i] == 7:
                    total_manhattan += 0
                if state[i] == 8:
                    total_manhattan += 1
                if state[i] == 0:
                    total_manhattan += 2
            if i == 7:
                if state[i] == 1:
                    total_manhattan += 3
                if state[i] == 2:
                    total_manhattan += 2
                if state[i] == 3:
                    total_manhattan += 3
                if state[i] == 4:
                    total_manhattan += 2
                if state[i] == 5:
                    total_manhattan += 1
                if state[i] == 6:
                    total_manhattan += 2
                if state[i] == 7:
                    total_manhattan += 1
                if state[i] == 8:
                    total_manhattan += 0
                if state[i] == 0:
                    total_manhattan += 1
            if i == 8:
                if state[i] == 1:
                    total_manhattan += 4
                if state[i] == 2:
                    total_manhattan += 3
                if state[i] == 3:
                    total_manhattan += 2
                if state[i] == 4:
                    total_manhattan += 3
                if state[i] == 5:
                    total_manhattan += 2
                if state[i] == 6:
                    total_manhattan += 1
                if state[i] == 7:
                    total_manhattan += 2
                if state[i] == 8:
                    total_manhattan += 1
                if state[i] == 0:
                    total_manhattan += 0

        return total_manhattan

#########################################################################################
# ALL CODE BELOW WRITTEN FOR COM S 472 Lab 1 Assignment
# @author Ryan Herren

# Displays board in 3x3 format
def display_board(board):
    count = 0
    s = ""
    for i in board:
        if count % 3 == 0:
            s += "\n"
        s += (" " + str(i) + " ")
        count += 1
    s += "\n"
    return s


# Reads input from file
def read_board(input):
    with open(input, 'r') as f:
        lines = f.readlines()
    puzzle_array = []
    for line in lines:
        puzzle_array.extend([i for i in line.strip().split()])
    return puzzle_array


# RUN
def main(argv):
    # READ INPUT FROM COMMAND LINE
    inputfile = ''
    outputfile = ''
    opts, args = getopt.getopt(argv, "hi:o:", ["fPath=", "alg="])
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -alg <algorithm>')
            sys.exit()
        elif opt in ("-i", "--fPath"):
            inputfile = arg
        elif opt in ("-alg", "--alg"):
            alg = arg
    print('Input file is', inputfile)
    print('Algorithm is', alg)
    # READ FROM SPECIFIED FILE
    new_in = "tests_472/" + inputfile
    temp = read_board(new_in)
    initial = ()
    for i in temp:
        if i == "_":
            initial = initial + ('0',)
        else:
            initial = initial + (i,)

    # FORMAT PUZZLE
    l1 = []
    for i in initial:
        l1.append(int(i))

    # INITIALIZE PUZZLES
    # Puzzle is used for BFS, IDS, and H1
    # Puzzle2 is used for H2
    # Puzzle3 is used for H3
    puzzle = EightPuzzle(tuple(l1))
    puzzle2 = EightPuzzle2(tuple(l1))
    puzzle3 = EightPuzzle3(tuple(l1))

    # CHOOSE ALGORITHM
    if alg in ("BFS", "bfs"):
        # CHECK SOLVABILITY
        if not EightPuzzle.check_solvability(puzzle, puzzle.goal):
            print("The inputted puzzle is not solvable: ", display_board(initial))
            return 0
        else:
            print("Puzzle is solvable... begin solving")
        start_time = time.time()
        end_state = breadth_first_graph_search(puzzle)
        print("Total time taken (seconds):", time.time() - start_time)
        print("Path length:", len(end_state.solution()))
        print(end_state.solution())
        print()
    elif alg in ("IDS", "ids"):
        # CHECK SOLVABILITY
        if not EightPuzzle.check_solvability(puzzle, puzzle.goal):
            print("The inputted puzzle is not solvable: ", display_board(initial))
            return 0
        else:
            print("Puzzle is solvable... begin solving")
        start_time = time.time()
        end_state = iterative_deepening_search(puzzle)
        print("Total time taken (seconds):", time.time() - start_time)
        print("Path length:", len(end_state.solution()))
        print(end_state.solution())
        print()
    elif alg in ("H1", "h1"):
        # CHECK SOLVABILITY
        if not EightPuzzle.check_solvability(puzzle, puzzle.goal):
            print("The inputted puzzle is not solvable: ", display_board(initial))
            return 0
        else:
            print("Puzzle is solvable... begin solving")
        start_time = time.time()
        end_state = astar_search(puzzle)
        print("Total time taken (seconds):", time.time() - start_time)
        print("Path length:", len(end_state.solution()))
        print(end_state.solution())
        print()
    elif alg in ("H2", "h2"):
        # CHECK SOLVABILITY
        if not EightPuzzle.check_solvability(puzzle2, puzzle2.goal):
            print("The inputted puzzle is not solvable: ", display_board(initial))
            return 0
        else:
            print("Puzzle is solvable... begin solving")
        start_time = time.time()
        end_state = astar_search(puzzle2)
        print("Total time taken (seconds):", time.time() - start_time)
        print("Path length:", len(end_state.solution()))
        print(end_state.solution())
        print()
    elif alg in ("H3", "h3"):
        # CHECK SOLVABILITY
        if not EightPuzzle.check_solvability(puzzle3, puzzle3.goal):
            print("The inputted puzzle is not solvable: ", display_board(initial))
            return 0
        else:
            print("Puzzle is solvable... begin solving")
        start_time = time.time()
        end_state = astar_search(puzzle3)
        print("Total time taken (seconds):", time.time() - start_time)
        print("Path length:", len(end_state.solution()))
        print(end_state.solution())
        print()

    return "fin"


if __name__ == "__main__":
    main(sys.argv[1:])
