import random
import numpy as np


VALUE_DICT = {0: -1, 1: -100, 2: -1, 3: 100}
DIRECTIONS = {0: "up", 1: "right", 2: "left", 3: "down"}

def dfs(graph, start, visited=None):
    if visited is None:
        found = False
        visited = set()

    if start in visited:
        return False

    row, column = start
    if graph[row][column] == 3:
        return True
    if graph[row][column] == 1:
        return False
    visited.add(start)

    if row > 0:
        if dfs(graph, (row - 1, column), visited):
            return True
    if row < len(graph) - 1:
        if dfs(graph, (row + 1, column), visited):
            return True
    if column > 0:
        if dfs(graph, (row, column - 1), visited):
            return True
    if column < len(graph[0]) - 1:
        if dfs(graph, (row, column + 1), visited):
            return True

    return False

def is_valid_location(row, column, board_size):
    if row < 0:
        return False
    if row > board_size - 1:
        return False
    if column < 0:
        return False
    if column > board_size - 1:
        return False
    return True

def generate_board(seed, n, hole_prob):
    random.seed(seed)
    start_coord = (0, 0)
    board = [[0]]
    while not dfs(board, start_coord):
        board = [[0 for _ in range(n)] for __ in range(n)]
        start = random.randrange(0, np.square(n))
        end = random.randrange(0, np.square(n))
        while end == start:
            end = random.randrange(0, np.square(n))
        row = start // n
        column = start % n
        board[row][column]  = 2
        row = end // n
        column = end % n
        board[row][column]  = 3

        for row in range(n):
            for column in range(n):
                if random.random() < hole_prob and board[row][column] == 0:
                    board[row][column] = 1

        start_coord = (start // n, start % n)
    return board

def is_terminal(board, row, column):
    if not is_valid_location(row, column, len(board)):
        return True
    value = VALUE_DICT[board[row][column]]
    if value == 100 or value == -100:
        return True
    return False

def choose_action(row, column, q_values, epsilon):
  if np.random.random() < epsilon:
    return np.argmax(q_values[row, column])
  else: #choose a random action
    return np.random.randint(4)

def next_location(current_row, current_column, direction):
  new_row_index = current_row
  new_column_index = current_column
  if DIRECTIONS[direction] == 'up':
    new_row_index -= 1
  elif DIRECTIONS[direction] == 'right':
    new_column_index += 1
  elif DIRECTIONS[direction] == 'down':
    new_row_index += 1
  elif DIRECTIONS[direction] == 'left':
    new_column_index -= 1
  return new_row_index, new_column_index

def run_random(board):
    row = 0
    column = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 2:
                row = i
                column = j

    while not is_terminal(board, row, column):
        row, column = next_location(row, column, np.random.randint(4))

    if row < len(board) and column < len(board) and row >= 0 and column >= 0:
        if board[row][column] == 3:
            return True
    return False

def run_q(board, q_values, epsilon, discount_factor, learning_rate):
    row = 0
    column = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 2:
                row = i
                column = j

    while not is_terminal(board, row, column):
        action_index = choose_action(row, column, q_values, epsilon)

        old_row, old_column = row, column
        row, column = next_location(row, column, action_index)

        if not is_valid_location(row, column, len(board)):
            reward = -100
        else:
            reward = VALUE_DICT[board[row][column]]

        old_q_value = q_values[old_row, old_column, action_index]
        if column < 0:
            temporal_difference = reward + (discount_factor * np.max(q_values[row, column + 1])) - old_q_value
        elif column > len(board) - 1:
            temporal_difference = reward + (discount_factor * np.max(q_values[row, column - 1])) - old_q_value
        elif row < 0:
            temporal_difference = reward + (discount_factor * np.max(q_values[row + 1, column])) - old_q_value
        elif row > len(board) - 1:
            temporal_difference = reward + (discount_factor * np.max(q_values[row - 1, column])) - old_q_value
        else:
            temporal_difference = reward + (discount_factor * np.max(q_values[row, column])) - old_q_value


        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row, old_column, action_index] = new_q_value

    if row < len(board) and column < len(board) and row >= 0 and column >= 0:
        if board[row][column] == 3:
            return True

    return False