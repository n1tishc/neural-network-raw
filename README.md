# Playground

Local playground project for experimenting with a neural network built from scratch in Python and NumPy.

## What’s inside

- `neural-network/` — main neural network implementation
- `neural-network/playground/` — local Streamlit app to interactively test features
- `nn-from-scratch-plan.md` — original implementation plan

## Features

The neural network project includes:
- Dense feedforward layers
- Activations: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, Linear
- Losses: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- Optimizers: SGD, SGD with Momentum, RMSprop, Adam
- Training, evaluation, metrics, and visualizations

The playground app lets you:
- Choose datasets like XOR, Spiral, and Regression
- Configure network depth, width, activations, optimizer, and learning rate
- Train interactively
- Visualize loss and decision boundaries
- Observe how architecture and hyperparameters affect results

## Project structure

```text
playground/
├── neural-network/
│   ├── src/
│   ├── tests/
│   ├── examples/
│   ├── playground/
│   │   ├── app.py
│   │   └── README.md
│   ├── requirements.txt
│   └── README.md
├── nn-from-scratch-plan.md
└── README.md
```

## Setup

```bash
cd neural-network
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the playground app

```bash
cd neural-network/playground
streamlit run app.py
```

Open:
- `http://localhost:8501`

## Run examples

```bash
cd neural-network/examples
python xor_demo.py
python spiral_classifier.py
python mnist_classifier.py
```

## Run tests

```bash
cd neural-network/tests
python test_activations.py
python test_layers.py
python test_losses.py
python test_network.py
```

## Notes

- The web app is intended for local experimentation.
- If imports fail, run commands from inside `neural-network/` or activate the virtual environment first.
- The main implementation README is in `neural-network/README.md`.
