# FIDE-Google-Efficient-Chess-AI-Challenge

An efficient chess AI agent for the FIDE & Google Kaggle competition.

https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge

![Competition Overview](https://storage.googleapis.com/kagglesdsdata/competitions/86524/9818394/Screenshot%202024-10-09%20at%2010.45.28AM.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20241205%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241205T035010Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=be51d87f36061bdfe0fd1cf76e3422242047d79a6f07c770e0a557b6e66fc5e90c06ea7b8258365a9eb5cd4242db90018ed2ae5e6b699e53b98a1eb27910e5940522c1120356e035d596a7c63df6b6307e030fd251c515eaa9e4acc1804a6bb43e9c27aed2fd72b81a32523793e7e874ba27a5c0d96f0b965e7ebee2d0b6c6dd223df8d08648aa8ff2b14a4a2777494e7f169a23d2f77dd79bce324a83241bf1bd2fa17fe467b89b9382e36c05d62c33bdea46edd13a601bc7cea358ce7a1c30e4ba08edfffa825c8a106ac73141f898827264d80cbd019f7673663f1b8c91ba39f05a15fdcd843f083e153f12b88ad9f2eb4152d8b1bf2507ce216406faef6a)

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