from q_learn import run_random, run_q, generate_board
import numpy as np

SEED = 5
N = 8
EPSILON = 0.5
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.5
CYCLES = 100

board = generate_board(SEED, N, 0.5)
for row in board:
    print(row)
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