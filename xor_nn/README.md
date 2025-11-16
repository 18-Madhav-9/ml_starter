# XOR Neural Network

A minimal feedforward neural network built from scratch in **standard C++17** that learns the XOR function using backpropagation and gradient descent.

---

## Architecture

```
Input (2)  →  Hidden (2)  →  Output (1)
           W1, B1        W2, B2
           sigmoid       sigmoid
```

---

## What it does

- Learns XOR truth table
- Uses sigmoid activation
- Trains via backpropagation
- Optimizes using gradient descent (MSE loss)

---

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

./xor_nn