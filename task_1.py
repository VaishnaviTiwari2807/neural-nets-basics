import numpy as np
import argparse

def main(input_file, output_file):
    # Load input values from file
    with open(input_file, 'r') as f:
        x = float(f.readline().strip())
        y = float(f.readline().strip())
        w1 = float(f.readline().strip())
        w2 = float(f.readline().strip())
        w3 = float(f.readline().strip())
        activation = f.readline().strip()

    # Define activation function
    if activation == 'Logistic':
        activation_fn = lambda x: 1 / (1 + np.exp(-x))
    elif activation == 'Tanh':
        activation_fn = np.tanh
    elif activation == 'Relu':
        activation_fn = lambda x: np.maximum(0, x)
    else:
        raise ValueError('Invalid activation function: {}'.format(activation))

    # Calculate forward pass
    h1 = activation_fn(x * w1)
    h2 = activation_fn(h1 * w2)
    y_pred = activation_fn(w3 * h2)

    # Calculate loss and derivatives
    loss = (y_pred - y) ** 2
    dL_dw1 = y_pred(1-y_pred)(y-y_pred)
    dL_dw2 = h2 * (1-h2) * w3 * dL_dw1
    dL_dw3 = h1 * (1-h1)

    # Save output to file
    with open(output_file, 'w') as f:
        f.write(str(dL_dw1) + '\n')
        f.write(str(dL_dw2) + '\n')
        f.write(str(dL_dw3) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='path to input file')
    parser.add_argument('output_file', type=str, help='path to output file')
    args = parser.parse_args()

    main(args.input_file, args.output_file)
