# Neural Network Playground

Interactive webapp to experiment with the neural network implementation and visualize results.

## Features

- **Dataset Selection**: XOR, Spiral (2/3 classes), Regression
- **Network Architecture**: Configure hidden layers, sizes, activations
- **Training**: Choose optimizer, learning rate, epochs, batch size
- **Visualization**: 
  - Live loss curves during training
  - Decision boundary plots for 2D data
  - Weight statistics

## Usage

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run the playground
cd playground
streamlit run app.py
```

Open browser at `http://localhost:8501`

## Experiment Ideas

1. **XOR Problem**: 2 hidden layers (8-4), ReLU, SGD, lr=0.1
2. **Spiral 3-class**: 3 hidden layers (32-16-8), ReLU, Adam, lr=0.001
3. **Impact of Activation**: Compare ReLU vs Sigmoid on spiral data
4. **Optimizer Comparison**: SGD vs Adam on same problem
5. **Depth vs Width**: Compare 1 layer (128) vs 4 layers (32-32-16-8)
