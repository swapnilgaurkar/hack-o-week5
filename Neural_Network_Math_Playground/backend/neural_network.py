import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def backprop_demo(x, w1, w2, target, learning_rate):
    # One input -> one hidden neuron -> one output neuron
    hidden_raw = x * w1
    hidden = sigmoid(hidden_raw)

    output_raw = hidden * w2
    output = sigmoid(output_raw)

    # Mean squared error for this tiny demonstration
    error = 0.5 * (target - output) ** 2

    # Backpropagation using chain rule
    d_error_d_output = output - target
    d_output_d_raw = output * (1 - output)
    d_raw_d_hidden = w2
    d_hidden_d_raw = hidden * (1 - hidden)

    d_error_d_w2 = d_error_d_output * d_output_d_raw * hidden
    d_error_d_w1 = (
        d_error_d_output
        * d_output_d_raw
        * d_raw_d_hidden
        * d_hidden_d_raw
        * x
    )

    new_w1 = w1 - learning_rate * d_error_d_w1
    new_w2 = w2 - learning_rate * d_error_d_w2

    return {
        "hidden": hidden,
        "output": output,
        "error": error,
        "grad_w1": d_error_d_w1,
        "grad_w2": d_error_d_w2,
        "new_w1": new_w1,
        "new_w2": new_w2
    }
