# FIDE-Google-Efficient-Chess-AI-Challenge

An efficient chess AI agent for the FIDE & Google Kaggle competition.

https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge

## Features

- Resource-efficient chess engine (< 5 MiB RAM, single 2.20GHz CPU core)
- Minimax search with alpha-beta pruning
- Iterative deepening with time management
- Basic evaluation function including:
  - Material balance
  - Piece-square tables
  - Mobility evaluation

## Requirements

- Python 3.7+
- python-chess
- numpy

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The agent can be used directly with the Kaggle competition framework. The main entry point is the `agent()` function in `main.py`.

## Implementation Details

The chess engine implements several key features:

1. **Search Algorithm**: Minimax with alpha-beta pruning and iterative deepening
2. **Evaluation Function**: 
   - Material counting
   - Basic positional evaluation using piece-square tables
   - Mobility evaluation
3. **Time Management**: Conservative time usage with early stopping

## Competition Constraints

This implementation adheres to the competition constraints:
- 5 MiB RAM limit
- Single 2.20GHz CPU core
- 64KiB compressed submission size limit
- 10s with 0.1s Simple Delay time control