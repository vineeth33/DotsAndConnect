from flask import Flask, render_template, jsonify, request, send_from_directory
import math
from typing import List, Tuple, Optional
import random
import time

GRID_SIZE = 5
MINIMAX_DEPTH = 3

app = Flask(__name__)


class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = [[0 for _ in range(GRID_SIZE + 1)]
                     for _ in range(GRID_SIZE + 1)]
        self.boxes = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.player_turn = 0  # 0: human, 1: AI
        self.scores = [0, 0]
        self.extra_turn = False
        self.last_move = None
        self.ai_difficulty = "medium"  # New attribute for AI difficulty

    def evaluate_position(self) -> float:
        if self.is_game_over():
            if self.scores[1] > self.scores[0]:
                return 1.0
            elif self.scores[0] > self.scores[1]:
                return -1.0
            return 0.0

        ai_potential = 0
        human_potential = 0
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.boxes[row][col] == 0:
                    sides = self.count_completed_sides(row, col)
                    if sides == 3:
                        if self.player_turn == 1:
                            ai_potential += 1
                        else:
                            human_potential += 1
                    elif sides == 2:
                        if self.player_turn == 1:
                            ai_potential += 0.5
                        else:
                            human_potential += 0.5

        score = (self.scores[1] - self.scores[0]) / (GRID_SIZE * GRID_SIZE)
        potential_score = (ai_potential - human_potential) / \
            (GRID_SIZE * GRID_SIZE)

        return score * 0.7 + potential_score * 0.3

    def count_completed_sides(self, row: int, col: int) -> int:
        count = 0
        if self.grid[row][col] & 1:
            count += 1
        if self.grid[row + 1][col] & 1:
            count += 1
        if self.grid[row][col] & 2:
            count += 1
        if self.grid[row][col + 1] & 2:
            count += 1
        return count

    def minimax(self, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[float, Optional[Tuple]]:
        if depth == 0 or self.is_game_over():
            return self.evaluate_position(), None

        moves = self.get_available_moves()
        if not moves:
            return self.evaluate_position(), None

        best_move = None
        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                old_grid = [row[:] for row in self.grid]
                old_boxes = [row[:] for row in self.boxes]
                old_scores = self.scores[:]
                old_turn = self.player_turn
                old_extra = self.extra_turn

                self.make_move(*move)
                next_max = self.extra_turn == maximizing_player

                eval_score, _ = self.minimax(depth - 1, alpha, beta, next_max)

                self.grid = [row[:] for row in old_grid]
                self.boxes = [row[:] for row in old_boxes]
                self.scores = old_scores[:]
                self.player_turn = old_turn
                self.extra_turn = old_extra

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in moves:
                old_grid = [row[:] for row in self.grid]
                old_boxes = [row[:] for row in self.boxes]
                old_scores = self.scores[:]
                old_turn = self.player_turn
                old_extra = self.extra_turn

                self.make_move(*move)
                next_max = self.extra_turn == maximizing_player

                eval_score, _ = self.minimax(depth - 1, alpha, beta, next_max)

                self.grid = [row[:] for row in old_grid]
                self.boxes = [row[:] for row in old_boxes]
                self.scores = old_scores[:]
                self.player_turn = old_turn
                self.extra_turn = old_extra

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def get_best_move(self) -> Optional[Tuple]:
        start_time = time.time()

        if self.ai_difficulty == "easy":
            move = self.get_random_move()
        elif self.ai_difficulty == "medium":
            _, move = self.minimax(2, float('-inf'), float('inf'), True)
        else:  # hard
            _, move = self.minimax(
                MINIMAX_DEPTH, float('-inf'), float('inf'), True)

        print(f"AI move took {time.time() - start_time:.2f} seconds")
        return move

    def get_random_move(self) -> Optional[Tuple]:
        available_moves = self.get_available_moves()
        return random.choice(available_moves) if available_moves else None

    def make_move(self, row: int, col: int, is_horizontal: bool) -> bool:
        if not self.is_valid_move(row, col, is_horizontal):
            return False

        if is_horizontal:
            self.grid[row][col] |= 1
        else:
            self.grid[row][col] |= 2

        self.last_move = (row, col, is_horizontal)

        completed_box = False
        self.extra_turn = False

        if is_horizontal:
            if row > 0:
                completed_box |= self.check_box(row - 1, col)
            if row < GRID_SIZE:
                completed_box |= self.check_box(row, col)
        else:
            if col > 0:
                completed_box |= self.check_box(row, col - 1)
            if col < GRID_SIZE:
                completed_box |= self.check_box(row, col)

        if not completed_box:
            self.player_turn = 1 - self.player_turn
        else:
            self.extra_turn = True

        return True

    def get_available_moves(self) -> List[Tuple]:
        moves = []
        for row in range(GRID_SIZE + 1):
            for col in range(GRID_SIZE):
                if not (self.grid[row][col] & 1):
                    moves.append((row, col, True))
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE + 1):
                if not (self.grid[row][col] & 2):
                    moves.append((row, col, False))
        return moves

    def is_valid_move(self, row: int, col: int, is_horizontal: bool) -> bool:
        if row < 0 or col < 0:
            return False

        if is_horizontal:
            if row > GRID_SIZE or col >= GRID_SIZE:
                return False
            return not (self.grid[row][col] & 1)
        else:
            if row >= GRID_SIZE or col > GRID_SIZE:
                return False
            return not (self.grid[row][col] & 2)

    def check_box(self, row: int, col: int) -> bool:
        if row < 0 or col < 0 or row >= GRID_SIZE or col >= GRID_SIZE:
            return False

        if (self.grid[row][col] & 1 and
            self.grid[row + 1][col] & 1 and
            self.grid[row][col] & 2 and
                self.grid[row][col + 1] & 2):

            if self.boxes[row][col] == 0:
                self.boxes[row][col] = self.player_turn + 1
                self.scores[self.player_turn] += 1
                self.extra_turn = True
                return True
        return False

    def is_game_over(self) -> bool:
        return sum(self.scores) == GRID_SIZE * GRID_SIZE

    def ai_play(self) -> None:
        while self.player_turn == 1 and not self.is_game_over():
            move = self.get_best_move()
            if move:
                row, col, is_horizontal = move
                self.make_move(row, col, is_horizontal)
                if not self.extra_turn:
                    break
            else:
                break


game = Game()


@app.route('/')
def index():
    return render_template('index.html', grid_size=GRID_SIZE)


@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    row, col, is_horizontal = data['row'], data['col'], data['isHorizontal']

    if game.make_move(row, col, is_horizontal):
        if not game.is_game_over() and game.player_turn == 1:
            game.ai_play()

    return jsonify({
        'grid': game.grid,
        'boxes': game.boxes,
        'scores': game.scores,
        'currentPlayer': 'human' if game.player_turn == 0 else 'ai',
        'gameOver': game.is_game_over(),
        'lastMove': game.last_move
    })


@app.route('/reset_game', methods=['POST'])
def reset_game():
    game.reset()
    return jsonify({'message': 'Game reset successfully'})


@app.route('/set_difficulty', methods=['POST'])
def set_difficulty():
    data = request.json
    difficulty = data['difficulty']
    if difficulty in ["easy", "medium", "hard"]:
        game.ai_difficulty = difficulty
        return jsonify({'message': f'AI difficulty set to {difficulty}'})
    else:
        return jsonify({'error': 'Invalid difficulty level'}), 400


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
