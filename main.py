from Connect4 import GameFinishedException, Connect4, RandomPlayer, MiniMaxPlayer, check_win, add_to_column
from time import process_time
ATTEMPTS = 200
HEIGHT = 5
WIDTH = 5
DEPTH = 5


def run_game():
    playing = True
    player1 = MiniMaxPlayer('Player 1', DEPTH)
    player2 = MiniMaxPlayer('Player 2', DEPTH)
    game = Connect4(player1, player2, HEIGHT, WIDTH)
    game_result = None
    while(playing):
        try:
            game.player_move(True)
            check_win(game)
        except GameFinishedException as e:
            game_result = e.result
            break

        try:
            game.player_move(False)
            check_win(game)
        except GameFinishedException as e:
            game_result = e.result
            break

    return game_result, game

def run_game_display():
    playing = True
    player1 = RandomPlayer('Player 1', DEPTH)
    player2 = RandomPlayer('Player 2', DEPTH)
    game = Connect4(player1, player2, HEIGHT, WIDTH)
    game_result = None
    round = 1
    while(playing):
        print(f'Round {round}')
        print(f'{player1.name} move')
        round += 1
        try:
            game.player_move(True)
            check_win(game)
        except GameFinishedException as e:
            game_result = e.result
            break
        game.display()

        print(f'{player2.name} move')
        try:
            game.player_move(False)
            check_win(game)
        except GameFinishedException as e:
            game_result = e.result
            break
        game.display()

    game.display()
    game.finish_game(game_result)
    return game_result, game


def main():
    wins = []
    print(f'Attempts: {ATTEMPTS}')
    start_time = process_time()
    for i in range(ATTEMPTS):
        result, game = run_game()
        wins.append(result)
    stop_time = process_time()
    print(f'{game.player1.name} wins: {wins.count(True)}')
    print(f'Draws: {wins.count(None)}')
    print(f'{game.player2.name} wins: {wins.count(False)}')
    print(f'Total time taken: {round(stop_time - start_time, 2)}')
    # run_game_display()


if __name__ == "__main__":
    main()