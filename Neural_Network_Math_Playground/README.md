# Neural Network Math Playground

An educational mini-project that demonstrates the mathematical foundations of neural networks.

## Topics covered

### Linear Algebra
- Vectors
- Vector operations
- Dot product
- Matrices
- Matrix multiplication
- Matrix-vector multiplication
- Eigenvalues and eigenvectors (intuition level)

### Calculus
- Derivatives
- Gradient
- Chain rule
- Backpropagation intuition

## Technology

- Frontend: HTML, CSS, JavaScript
- Backend: Python + Flask
- Mathematics: NumPy + SymPy

## How to run on macOS

### 1. Open Terminal

Go inside the project folder:

```bash
cd /path/to/Neural_Network_Math_Playground
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate it

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 5. Start the server

```bash
python3 backend/app.py
```

You should see Flask running on:

http://127.0.0.1:5000

### 6. Open the project

Open this in Chrome:

http://127.0.0.1:5000

## Stop the server

In Terminal:

```text
Control + C
```

To leave the virtual environment:

```bash
deactivate
```

## Project structure

```text
Neural_Network_Math_Playground/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── backend/
│   ├── app.py
│   ├── vectors.py
│   ├── matrices.py
│   ├── gradients.py
│   └── neural_network.py
├── requirements.txt
└── README.md
```

## Demo flow for presentation

1. Enter two vectors and calculate their dot product.
2. Show matrix multiplication and matrix-vector transformation.
3. Explain eigenvalues using the diagonal matrix example.
4. Change x and observe the derivative/slope.
5. Change x and y to observe the gradient.
6. Explain the chain rule.
7. Run the backpropagation demo and show how gradients change the weights.

## Important

This is an educational visualization project. The backpropagation example intentionally uses a tiny one-hidden-neuron network so the mathematics can be followed manually.
