import random
import numpy as np

N = 8

def generate_board(seed, n, hole_prob):
    board = [[0 for _ in range(n)] for __ in range(n)]
    random.seed(seed)
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

    return board

print(generate_board(10, 8, 0.5))


