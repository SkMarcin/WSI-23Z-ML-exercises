from q_learn import run_random, run_q, generate_board, dfs, get_shortest_path
import numpy as np

SEED = 38
N = 8
EPSILON = 0.5
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.5
CYCLES = 100000

board = [[2, 1], [1, 3]]
while not dfs(board, (0, 0)):
    board = generate_board(SEED, N, 0.5)
    board[0][0] = 2
    board[N - 1][N - 1] = 3
    SEED += 1
for row in board:
    print(row)
print (SEED)
q_values = np.zeros((N, N, 4))
q_learn = 0
random = 0
for _ in range(CYCLES):
    if run_q(board, q_values, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE):
        q_learn += 1
    if run_random(board):
        random += 1

print("hello")
print(q_learn)
print(random)
for coordinate in get_shortest_path(board, q_values, 0, 0):
    board[coordinate[0]][coordinate[1]] = '-'

for row in board:
    print('[', end="")
    for index, value in enumerate(row):
        if index != N - 1:
            print(f'{value}, ', end="")
        else:
            print(f'{value}', end="")
    print(']')