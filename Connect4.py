from random import choice
import copy

class GameFinishedException(Exception):
    def __init__(self, result) -> None:
        super().__init__()
        self.result = result


class InvalidSizeException(Exception):
    def __init__(self) -> None:
        super().__init__()


class Player:
    def __init__(self, name, depth) -> None:
        self.name = name
        self.depth = depth


class RandomPlayer(Player):
    def get_move(self, game, first_player, depth):
        empty_columns = get_empty_columns(game)
        if not empty_columns:
            raise GameFinishedException(None)
        selected = choice(empty_columns)
        add_to_column(game.board, game.height, selected, first_player)


class MiniMaxPlayer(Player):
    def get_move(self, game, first_player, depth):
        column = self.get_minimax(game, depth, first_player)[1]
        # print(f'column - {column}')
        add_to_column(game.board, game.height, column, first_player)

    def get_minimax(self, game, depth, max_move):
        winner = None
        try:
            check_win(game)
        except GameFinishedException as e:
            winner = e.result
        if winner != None:
            return winner, None
        empty_columns = get_empty_columns(game)
        if depth == 0 or not empty_columns:
            return None, None
        original_board = copy.deepcopy(game.board)
        results = {}
        for i in empty_columns:
            add_to_column(game.board, game.height, i, max_move)
            results[i] = self.get_minimax(game, depth - 1, not max_move)[0]
            game.board = copy.deepcopy(original_board)

        if max_move:
            columns = []
            for key in results:
                if results[key] == True:
                    columns.append(key)
            if columns:
                return True, choice(columns)
            for key in results:
                if results[key] == None:
                    columns.append(key)
            if columns:
                return None, choice(columns)
            for key in results:
                if results[key] == False:
                    columns.append(key)
            return False, choice(columns)

        else:
            columns = []
            for key in results:
                if results[key] == False:
                    columns.append(key)
            if columns:
                return False, choice(columns)
            for key in results:
                if results[key] == None:
                    columns.append(key)
            if columns:
                return None, choice(columns)
            for key in results:
                if results[key] == True:
                    columns.append(key)
            return True, choice(columns)


class Connect4:
    def __init__(self, player1, player2, height, width):
        self.player1 = player1
        self.player2 = player2
        if width >= 4 and height >= 5:
            self.width = width
            self.height = height
        else:
            raise InvalidSizeException
        self.board = [[None for j in range(width)] for i in range(height)]

    def player_move(self, first_player):
        if first_player == True:
            self.player1.get_move(self, first_player, self.player1.depth)
        else:
            self.player2.get_move(self, first_player, self.player2.depth)

    def finish_game(self, winner):
        if winner == True:
            print(f'{self.player1.name} win')
            return True
        elif winner == False:
            print(f'{self.player2.name} win')
            return False
        else:
            print(f'Game ended in draw')
            return None

    def display(self):
        for i in range(self.height):
            if (i != 0):
                print('-' * (6 * self.width + 1))
            print('|', end="")
            for j in range(self.width):
                if self.board[i][j] == None:
                    print(f'  0  |', end="")
                elif self.board[i][j] == True:
                    print(f'  1  |', end="")
                else:
                    print(f'  -1 |', end="")
            print(" ")
        print('\n')


def add_to_column(board, height, column, first_player):
    for i in range(height):
        if(board[height - i - 1][column]) == None:
            board[height - i - 1][column] = first_player
            break


def check_win(game):
        # check horizontal
        for row in range(game.height):
            for column in range(game.width - 3):
                if game.board[row][column] == game.board[row][column + 1] == game.board[row][column + 2] == game.board[row][column + 3] == True:
                    raise GameFinishedException(True)
                elif game.board[row][column] == game.board[row][column + 1] == game.board[row][column + 2] == game.board[row][column + 3] == False:
                    raise GameFinishedException(False)

        # check vertical
        for row in range(game.height - 3):
            for column in range(game.width):
                if game.board[row][column] == game.board[row + 1][column] == game.board[row + 2][column] == game.board[row + 3][column] == True:
                    raise GameFinishedException(True)
                elif game.board[row][column] == game.board[row + 1][column] == game.board[row + 2][column] == game.board[row + 3][column] == False:
                    raise GameFinishedException(False)

        # check diagonals
        for row in range(game.height - 3):
            for column in range(game.width - 3):
                if game.board[row][column] == game.board[row + 1][column + 1] == game.board[row + 2][column + 2] == game.board[row + 3][column + 3] == True:
                    raise GameFinishedException(True)
                elif game.board[row][column] == game.board[row + 1][column + 1] == game.board[row + 2][column + 2] == game.board[row + 3][column + 3] == False:
                    raise GameFinishedException(False)

        for row in range(game.height - 3):
            for column in range(game.width - 3):
                if game.board[row][column + 3] == game.board[row + 1][column + 2] == game.board[row + 2][column + 1] == game.board[row + 3][column] == True:
                    raise GameFinishedException(True)
                elif game.board[row][column + 3] == game.board[row + 1][column + 2] == game.board[row + 2][column + 1] == game.board[row + 3][column] == False:
                    raise GameFinishedException(False)

        if not get_empty_columns(game):
            raise GameFinishedException(None)

        return None


def get_empty_columns(game):
        top_layer = [i for i in range(game.width)]
        empty_columns = []
        for i in top_layer:
            if game.board[0][i] == None:
                empty_columns.append(i)
        return empty_columns


if __name__ == "__main__":
    game = Connect4(1, 1, 5, 4)
    # p1= RandomPlayer("xd")
    # p1.get_move(game)
    print(game.board)
    game.board = [[None, None, None, None], [1, -1, None, None], [1, 1, -1, None], [1, -1, 1, None], [-1, -1, 1, 1]]
    game.display()
    print(check_win(game))
