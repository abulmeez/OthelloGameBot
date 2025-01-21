# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """
  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.time_limit = 1.95
    self.node_count = 0
    self.depth_reached = 0

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    start_time = time.time()
    return_move = None # move we are going to return
    depth = 1
    self.node_count = 0
    self.depth_reached = 0

    try:
      while True:
        best_move_at_depth = None
        best_score_at_depth = float('-inf')

        valid_moves = self.order_moves(chess_board, get_valid_moves(chess_board, player), player, opponent)

        for move in valid_moves:
          if time.time() - start_time > self.time_limit:
            raise TimeoutError

          current_board = deepcopy(chess_board)
          execute_move(current_board, move, player)
          move_value = self.minimax(current_board, depth, maxx=False, player=player, opponent=opponent, alpha=float('-inf'), beta=float('inf'), start_time=start_time)

          if move_value > best_score_at_depth:
            best_score_at_depth = move_value
            best_move_at_depth = move

        if best_move_at_depth is not None:
          return_move = best_move_at_depth

        self.depth_reached = max(self.depth_reached, depth)
        depth += 1

    except TimeoutError:
      pass

    time_taken = time.time() - start_time
    print(f"My AI's turn took {time_taken} seconds. Nodes evaluated: {self.node_count}, Depth reached: {self.depth_reached}")

    if return_move is None: # just in case
      return get_valid_moves(chess_board, player)[0]

    return return_move

  def minimax(self, board, depth, maxx, player, opponent, alpha, beta, start_time):
    if time.time() - start_time > self.time_limit:
      raise TimeoutError

    self.node_count += 1
    self.depth_reached = max(self.depth_reached, depth)

    is_endgame, _, _ = check_endgame(board, player, opponent)

    if depth == 0 or is_endgame:
      return self.evaluation_function(board, player, opponent)

    if maxx:
      max_eval = float('-inf')
      valid_moves = self.order_moves(board, get_valid_moves(board, player), player, opponent)

      if not valid_moves:
        return self.minimax(board, depth - 1, False, player, opponent, alpha, beta, start_time=start_time)

      for move in valid_moves:
        if time.time() - start_time > self.time_limit:
          raise TimeoutError

        current_board = deepcopy(board)
        execute_move(current_board, move, player)
        eval = self.minimax(current_board, depth - 1, False, player, opponent, alpha, beta, start_time=start_time)

        max_eval = max(max_eval, eval)
        alpha = max(alpha, eval)
        if beta <= alpha:
          break

      return max_eval

    else:
      # Null-Move Pruning, asked chat and said this will imrpove pruning so we evaluate more
      if depth >= 3 and beta - alpha < 50:
        if time.time() - start_time > self.time_limit:
          raise TimeoutError
        eval = self.minimax(board, depth - 1 - 2, True, player, opponent, alpha, beta, start_time=start_time)
        if eval >= beta:
          return eval

      min_eval = float('inf')
      valid_moves = self.order_moves(board, get_valid_moves(board, opponent), opponent, player)

      if not valid_moves:
        return self.minimax(board, depth - 1, True, player, opponent, alpha, beta, start_time=start_time)

      for move in valid_moves:
        if time.time() - start_time > self.time_limit:
          raise TimeoutError

        current_board = deepcopy(board)
        execute_move(current_board, move, opponent)
        eval = self.minimax(current_board, depth - 1, True, player, opponent, alpha, beta, start_time=start_time)

        min_eval = min(min_eval, eval)
        beta = min(beta, eval)

        if beta <= alpha:
          break

      return min_eval

  def generate_positional_weights(self, board_size, game_phase):
    weights = np.zeros((board_size, board_size))

    # Base weight values
    corner_weight = 20
    edge_weight = 5
    x_square_weight = -10
    c_square_weight = -5
    inner_weight = 1

    # Adjust weights based on game phase
    if game_phase == 'EARLY_GAME':
        corner_weight *= 1.5
        edge_weight *= 1.2
        x_square_weight *= 1.2
        c_square_weight *= 1.1
        inner_weight *= 1.0
    elif game_phase == 'MID_GAME':
        pass  # Keep base weights
    elif game_phase == 'LATE_GAME':
        corner_weight *= 0.8
        edge_weight *= 0.9
        x_square_weight *= 0.9
        c_square_weight *= 0.95
        inner_weight *= 1.1

    # Define positions
    corners = [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]
    x_squares = self.get_x_squares(board_size)
    c_squares = self.get_c_squares(board_size)

    for x in range(board_size):
        for y in range(board_size):
            # Corners
            if (x, y) in corners:
                weights[x][y] = corner_weight
            # Edges (excluding corners)
            elif x == 0 or x == board_size - 1 or y == 0 or y == board_size - 1:
                weights[x][y] = edge_weight
            # X-squares
            elif (x, y) in x_squares:
                weights[x][y] = x_square_weight
            # C-squares
            elif (x, y) in c_squares:
                weights[x][y] = c_square_weight
            else:
                weights[x][y] = inner_weight

    return weights

  def get_x_squares(self, board_size):
    x_squares = []
    positions = [1, board_size - 2]
    for x in positions:
        for y in positions:
            x_squares.append((x, y))
    return x_squares

  def get_c_squares(self, board_size):
    c_squares = []
    positions = [1, board_size - 2]
    corners = [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]

    for pos in positions:
        c_squares.extend([
            (0, pos), (pos, 0),
            (board_size - 1, pos), (pos, board_size - 1)
        ])
    return c_squares

  def get_game_phase(self, board):
    total_discs = np.count_nonzero(board != 0)
    board_size = board.shape[0]
    max_discs = board_size * board_size
    if total_discs < max_discs * 0.25:
        return 'EARLY_GAME'
    elif total_discs < max_discs * 0.75:
        return 'MID_GAME'
    else:
        return 'LATE_GAME'

  def eval_mobility(self, board, player, opponent):
    player_moves = len(get_valid_moves(board, player))
    opponent_moves = len(get_valid_moves(board, opponent))
    total_moves = player_moves + opponent_moves
    if total_moves == 0:
        return 0
    return 100 * (player_moves - opponent_moves) / total_moves

  def eval_corner(self, board, player, opponent):
    board_size = board.shape[0]
    corners = [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]
    player_corners = sum(1 for corner in corners if board[corner] == player)
    opponent_corners = sum(1 for corner in corners if board[corner] == opponent)
    total_corners = player_corners + opponent_corners
    if total_corners == 0:
        return 0
    return 100 * (player_corners - opponent_corners) / total_corners

  def eval_disc_diff(self, board, player, opponent):
    player_discs = np.count_nonzero(board == player)
    opponent_discs = np.count_nonzero(board == opponent)
    total_discs = player_discs + opponent_discs
    if total_discs == 0:
        return 0
    return 100 * (player_discs - opponent_discs) / total_discs

  def eval_parity(self, board):
    total_discs = np.count_nonzero(board != 0)
    remaining_moves = board.size - total_discs
    return 1 if remaining_moves % 2 == 0 else -1

  def evaluation_function(self, board, player, opponent):
    game_phase = self.get_game_phase(board)
    board_size = board.shape[0]
    positional_weights = self.generate_positional_weights(board_size, game_phase)
    
    player_positions = (board == player)
    opponent_positions = (board == opponent)
    
    player_score = np.sum(positional_weights[player_positions])
    opponent_score = np.sum(positional_weights[opponent_positions])
    
    positional_score = player_score - opponent_score

    # Heuristics
    corner_score = self.eval_corner(board, player, opponent)
    mobility_score = self.eval_mobility(board, player, opponent)
    disc_diff_score = self.eval_disc_diff(board, player, opponent)
    parity_score = self.eval_parity(board)

    # Adjust weights based on game phase
    if game_phase == 'EARLY_GAME':
        total_score = (
            1000 * corner_score +
            50 * mobility_score +
            10 * positional_score
        )
    elif game_phase == 'MID_GAME':
        total_score = (
            1000 * corner_score +
            20 * mobility_score +
            10 * disc_diff_score +
            100 * parity_score +
            5 * positional_score
        )
    else:  # LATE_GAME
        total_score = (
            1000 * corner_score +
            100 * mobility_score +
            500 * disc_diff_score +
            500 * parity_score +
            2 * positional_score
        )

    return total_score

  def order_moves(self, board, moves, player, opponent):
    move_scores = []
    for move in moves:
        temp_board = board.copy()
        execute_move(temp_board, move, player)
        score = self.evaluation_function(temp_board, player, opponent)
        move_scores.append((score, move))
    move_scores.sort(reverse=True)
    ordered_moves = [move for score, move in move_scores]
    return ordered_moves
