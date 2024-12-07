"""
Author: Ilya (https://www.ilyaaa.com/)
Support: https://www.ilyaaa.com/support

FIDE-Google Efficient Chess AI Challenge Agent
This agent implements a resource-efficient chess AI within the competition constraints:
- 5 MiB RAM
- Single 2.20GHz CPU core
- 64KiB compressed size limit
"""

import chess
import numpy as np
import time
from typing import Dict, List, Optional, Tuple

class ChessAgent:
    def __init__(self):
        # Piece values for basic material evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Simple piece-square tables for positional evaluation
        self.pst = self._initialize_piece_square_tables()
        
    def _initialize_piece_square_tables(self) -> Dict:
        """Initialize basic piece-square tables for positional evaluation."""
        # Simple piece-square tables (to be optimized)
        pst = {
            chess.PAWN: np.array([
                0,  0,  0,  0,  0,  0,  0,  0,
                50, 50, 50, 50, 50, 50, 50, 50,
                10, 10, 20, 30, 30, 20, 10, 10,
                5,  5, 10, 25, 25, 10,  5,  5,
                0,  0,  0, 20, 20,  0,  0,  0,
                5, -5,-10,  0,  0,-10, -5,  5,
                5, 10, 10,-20,-20, 10, 10,  5,
                0,  0,  0,  0,  0,  0,  0,  0
            ]),
            chess.KNIGHT: np.array([
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

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate the current position.
        Returns a score from white's perspective.
        """
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        
        score = 0
        
        # Material evaluation
        for piece_type in self.piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
        
        # Basic mobility evaluation
        score += len(list(board.legal_moves)) * (1 if board.turn else -1)
        
        return score

    def get_best_move(self, board: chess.Board, depth: int = 3) -> Optional[chess.Move]:
        """Find the best move using minimax with alpha-beta pruning."""
        def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[chess.Move]]:
            if depth == 0 or board.is_game_over():
                return self.evaluate_position(board), None
            
            best_move = None
            if maximizing:
                max_eval = float('-inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval_score, _ = minimax(board, depth - 1, alpha, beta, False)
                    board.pop()
                    
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move = move
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
                return max_eval, best_move
            else:
                min_eval = float('inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval_score, _ = minimax(board, depth - 1, alpha, beta, True)
                    board.pop()
                    
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_move = move
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                return min_eval, best_move

        _, best_move = minimax(board, depth, float('-inf'), float('inf'), True)
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
    # Initialize board from FEN if provided
    board = chess.Board(obs.get('fen', chess.STARTING_FEN))
    
    # Create agent instance
    chess_agent = ChessAgent()
    
    # Get the best move with iterative deepening
    start_time = time.time()
    time_limit = 0.1  # Conservative time limit to ensure we don't timeout
    depth = 1
    best_move = None
    
    while time.time() - start_time < time_limit and depth <= 4:
        try:
            move = chess_agent.get_best_move(board, depth)
            if move:
                best_move = move
            depth += 1
        except Exception:
            break
    
    # Return the best move found
    return best_move.uci() if best_move else list(board.legal_moves)[0].uci()
