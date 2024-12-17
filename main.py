"""
FIDE-Google Efficient Chess AI Challenge Agent
This agent implements a resource-efficient chess AI within the competition constraints:
- 5 MiB RAM
- Single 2.20GHz CPU core
- 64KiB compressed size limit
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from chessnut import Game, Move, InvalidMove

class ChessAgent:
    def __init__(self):
        # Piece values for basic material evaluation
        self.piece_values = {
            'p': 100,   # pawn
            'n': 320,   # knight
            'b': 330,   # bishop
            'r': 500,   # rook
            'q': 900,   # queen
            'k': 20000  # king
        }
        
        # Simple piece-square tables for positional evaluation
        self.pst = self._initialize_piece_square_tables()
        
    def _initialize_piece_square_tables(self) -> Dict:
        """Initialize basic piece-square tables for positional evaluation."""
        pst = {
            'p': np.array([  # Pawn
                0,  0,  0,  0,  0,  0,  0,  0,
                50, 50, 50, 50, 50, 50, 50, 50,
                10, 10, 20, 30, 30, 20, 10, 10,
                5,  5, 10, 25, 25, 10,  5,  5,
                0,  0,  0, 20, 20,  0,  0,  0,
                5, -5,-10,  0,  0,-10, -5,  5,
                5, 10, 10,-20,-20, 10, 10,  5,
                0,  0,  0,  0,  0,  0,  0,  0
            ]),
            'n': np.array([  # Knight
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ])
        }
        return pst

    def evaluate_position(self, game: Game) -> float:
        """
        Evaluate the current position.
        Returns a score from white's perspective.
        """
        board = game.board
        score = 0
        
        # Material evaluation
        for i, piece in enumerate(board):
            if piece != ' ':
                is_white = piece.isupper()
                piece_type = piece.lower()
                value = self.piece_values[piece_type]
                score += value if is_white else -value
                
                # Add piece-square table bonus for pawns and knights
                if piece_type in self.pst:
                    square_value = self.pst[piece_type][i if is_white else 63-i]
                    score += square_value if is_white else -square_value
        
        # Basic mobility evaluation (simplified)
        moves = len(list(game.get_moves()))
        score += moves if game.state.player == 'w' else -moves
        
        # Penalize blocked center pawns and knights
        center_squares = [27, 28, 35, 36]  # e4, d4, e5, d5
        for square in center_squares:
            piece = board[square]
            if piece.lower() in ['p', 'n']:
                is_white = piece.isupper()
                score -= 10 if is_white else -10
        
        return score

    def get_best_move(self, game: Game, depth: int = 3) -> Optional[str]:
        """Find the best move using minimax with alpha-beta pruning."""
        def minimax(game: Game, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[str]]:
            if depth == 0:
                return self.evaluate_position(game), None
            
            moves = list(game.get_moves())
            if not moves:
                return -20000 if maximizing else 20000, None
            
            best_move = moves[0]
            if maximizing:
                max_eval = float('-inf')
                for move in moves:
                    try:
                        new_game = Game(game.get_fen())
                        new_game.apply_move(move)
                        eval_score, _ = minimax(new_game, depth - 1, alpha, beta, False)
                        
                        if eval_score > max_eval:
                            max_eval = eval_score
                            best_move = move
                        alpha = max(alpha, eval_score)
                        if beta <= alpha:
                            break
                    except InvalidMove:
                        continue
                return max_eval, best_move
            else:
                min_eval = float('inf')
                for move in moves:
                    try:
                        new_game = Game(game.get_fen())
                        new_game.apply_move(move)
                        eval_score, _ = minimax(new_game, depth - 1, alpha, beta, True)
                        
                        if eval_score < min_eval:
                            min_eval = eval_score
                            best_move = move
                        beta = min(beta, eval_score)
                        if beta <= alpha:
                            break
                    except InvalidMove:
                        continue
                return min_eval, best_move

        _, best_move = minimax(game, depth, float('-inf'), float('inf'), True)
        return best_move

def agent(obs, config):
    """
    Main agent function that will be called by the competition framework.
    Args:
        obs: Observation from the environment
        config: Configuration for the game
    Returns:
        move: A chess move in UCI format (e.g., 'e2e4')
    """
    # Initialize game from FEN if provided
    game = Game(obs.get('fen', Game.START_POS))
    
    # Create agent instance
    chess_agent = ChessAgent()
    
    # Get the best move with iterative deepening
    start_time = time.time()
    time_limit = 0.1  # Conservative time limit to ensure we don't timeout
    depth = 1
    best_move = None
    
    while time.time() - start_time < time_limit and depth <= 4:
        try:
            move = chess_agent.get_best_move(game, depth)
            if move:
                best_move = move
            depth += 1
        except Exception:
            break
    
    # Return the best move found or a random legal move
    if best_move:
        return best_move
    else:
        moves = list(game.get_moves())
        return moves[0] if moves else None
