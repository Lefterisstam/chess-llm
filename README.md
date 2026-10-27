# Chess LLM

A chess move suggestion system built on Mistral 7B. The model was fine-tuned with QLoRA on games
of strong players from the Lichess database, and Stockfish checks its suggestions before the final
move is chosen.

This is the code of my master's thesis, *Development of a Chess Move Suggestion System Based on
Large Language Models*.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lefterisstam/chess-llm-thesis/blob/main/notebooks/demo.ipynb)

## How it works

The model sees the position as text: the FEN, the last ten half-moves and the side to move.

```
### Instruction:
You are a chess grandmaster. Analyze the current position and suggest the best next move.
Respond ONLY with the move in UCI format (e.g., e2e4, g1f3, e1g1 for castling).

Position (FEN): r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3
Move history (last 5 moves): e2e4 e7e5 g1f3 b8c6 f1b5
Side to move: Black

### Response:
```

1. The model gives a probability to every legal move in the position, so it can never suggest an
   illegal move.
2. The ten most likely moves go to Stockfish, which searches each one at depth 6.
3. The move with the best Stockfish score is played.

## Results

Top-1 accuracy is how often the system plays the same move as the human player. Measured on 1,000
positions from the test set:

| Method | Base Mistral | Stage 1 | Final model |
|---|---|---|---|
| Greedy decoding + legal move filter | 7.4% | 20.5% | 27.4% |
| Beam search (5) + legal move filter | 10.0% | 25.9% | 33.2% |
| Beam search over legal moves only | 11.2% | 26.0% | 33.3% |
| Scoring every legal move | 10.0% | 25.8% | 33.8% |
| Scoring every legal move + Stockfish rerank | 28.5% | 42.9% | **48.2%** |

The last two methods always return a legal move. Stockfish alone at depth 6, choosing among all legal
moves, reaches 48.5%, so the final system matches the engine but does not beat it.

## Data

- Two months of Lichess games (November 2020 and June 2024), 167.6 million games.
- Kept games where both players are rated 2200 or more, the base time is at least 8 minutes, the
  game ended normally and lasted 20 to 150 moves: 163,252 games.
- Ten random positions per game, skipping the first six half-moves: 1,632,520 examples.
- Examples where the human move lost more than 50 centipawns (Stockfish, depth 12) were dropped:
  1,178,904 examples left. The human move is always kept as the answer.
- Split by game, so positions of one game never end up in two sets: 941,765 train, 118,515
  validation and 118,624 test examples.

## Training

QLoRA: Mistral 7B in 4-bit NF4, LoRA with rank 32 and alpha 16 on the attention projections
(27.3 million trainable parameters).

- **Stage 1:** 8,000 steps, learning rate 1e-4, batch 24.
- **Stage 2:** the stage 1 adapter trained for one full pass over the data with learning rate 5e-5.

The final adapter is on the Hugging Face Hub:
[Leftstam/chess-llm-mistral-7b-v2-lora](https://huggingface.co/Leftstam/chess-llm-mistral-7b-v2-lora).

## Usage

An NVIDIA GPU is needed, because the model is loaded in 4-bit with bitsandbytes. The easiest way is
the Colab notebook linked above. To run it yourself:

```bash
pip install -r requirements.txt
sudo apt-get install stockfish

python model/inference/play.py --fen "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3" --history "e2e4 e7e5 g1f3 b8c6 f1b5"
python web/app.py
```

To rebuild everything from scratch:

```bash
python data/scripts/download_lichess.py --months 2020-11 2024-06 --output-dir data/raw
python data/scripts/run_pipeline.py --raw-dir data/raw --work-dir data
python model/training/train.py
python model/training/train.py --continued --packing --per-device-train-batch-size 16
python evaluation/metrics.py --adapter Leftstam/chess-llm-mistral-7b-v2-lora --test-jsonl data/processed/test.jsonl --stockfish-rerank --stockfish /usr/games/stockfish --output results.json
```

The training paths in `model/config/training_config.py` point to Google Drive, as the training ran on
Colab.

## Project layout

```
data/scripts/      download, filter, convert to examples, Stockfish check, split
model/config/      LoRA, quantization and training settings
model/training/    training script
model/inference/   ways of getting a move out of the model, Stockfish rerank
evaluation/        accuracy, centipawn loss, per-phase analysis, full games
web/               Flask app with a board to play against the model
notebooks/         Colab notebooks for the data, the training and the demo
tests/             pytest tests, run with `pytest`
```
