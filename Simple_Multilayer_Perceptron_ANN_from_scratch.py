'''
Implementation of deep neural network, to make it capable of identifying any 5 bits with first and last bits as 1
NOTE: This ∂L/∂ŷ should be read like (Loss gradient w.r.t output from output layer). Likewise for others
'''

import numpy as np
from sklearn.metrics import precision_score

# Activation function used is sigmoid function
def sigmoid(x): # performs sigmoid function on all values at once in vector
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): # performs derivative on all values at once (outputs of sigmoid functions) in vector
    return x * (1 - x)

# Loss (Cost) function used is MSE (Mean Squared Error) => sum[(y - ŷ)^2] / n
def mean_squared_error(y_actual, y_predicted): # these arguments are vectors
    error = y_actual - y_predicted
    mse = np.mean(error ** 2)
    return mse # MSE is a scalar value here

# derivative of MSE is -2(y - ŷ) NOTE: typically we drop the constant -2 (since it's absorbed by the learning rate in gradient descent algorithm)
def mean_squared_error_derivative(y_actual, y_predicted):
    d_error = y_actual - y_predicted
    return d_error

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.1):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate

        # Initializing weights between each layer
        self.weights_input_hidden = np.random.rand(n_inputs, n_hidden)
        self.weights_hidden_output = np.random.rand(n_hidden, n_outputs)

        # Initializing biases for each layer
        self.bias_hidden = np.zeros((1, n_hidden))
        self.bias_output = np.zeros((1, n_outputs))

    def get_parameters(self):
        return [self.weights_input_hidden, self.weights_hidden_output, self.bias_hidden, self.bias_output]

    def forward_propagation(self, X):
        # input layer to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden # computes z_hidden = W_input_hidden * X + b_hidden
        self.hidden_output = sigmoid(self.hidden_input) # computes a_hidden = sigmoid(z_hidden)

        # hidden layer to output layer
        self.last_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output # computes z_out = W_hidden_out * a_hidden + b_out
        self.last_output = sigmoid(self.last_input) # computes a_out = sigmoid(z_out)

        return self.last_output # ŷ = a_out

    def backward_propagation(self, X, y):
        # getting the loss for calculating its gradient wrt each parameter (as done by mean_squared_error_derivative())
        # and for evaluating the loss of the network after each epoch
        loss = mean_squared_error(y, self.last_output)

        #--------------Chain Rule part---------------
        # Gradients for output layer
        d_loss = mean_squared_error_derivative(y, self.last_output) # computes ∂L/∂ŷ for output layer NOTE: ŷ is nothing but a_out
        d_output = d_loss * sigmoid_derivative(self.last_output) # computes ∂L/∂z_out = (∂L/∂ŷ) * (∂ŷ/∂z_out) for output layer

        # Error to propagate to hidden layer
        loss_hidden = d_output.dot(self.weights_hidden_output.T) # computes ∂L/∂a_hidden = (∂L/∂z_out) * (∂z_out/∂a_hidden)
        # instead of computing ∂z_out/∂a_hidden, we used (W_hidden_output)^T
        d_hidden = loss_hidden * sigmoid_derivative(self.hidden_output) # computes ∂L/∂z_hidden = (∂L/∂a_hidden) * (∂a_hidden/∂z_hidden)
        #--------------------------------------------

        #-----------Gradient Descent part------------
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate
        #--------------------------------------------
        # Return mean squared error for monitoring
        return loss

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            self.forward_propagation(X)
            loss = self.backward_propagation(X, y)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return self.get_parameters() # for seeing the learnt (optimized) weights and biases

    def predict(self, X):
        output = self.forward_propagation(X)
        return output


if __name__ == "__main__":
    train = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        #[0, 0, 0, 1, 0], for test
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        #[0, 0, 1, 0, 1], for test
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        #[0, 1, 0, 0, 0], for test
        #[0, 1, 0, 0, 1], for test
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1],
        #[1, 0, 0, 0, 0], for test
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 1, 1],
        #[1, 1, 0, 0, 0], for test
        [1, 1, 0, 0, 1],
        [1, 1, 0, 1, 0],
        #[1, 1, 0, 1, 1], for test
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
        #[1, 1, 1, 1, 1] for test
    ])

    # Condition: first bit AND last bit are both 1
    labels = np.array([1 if row[0] == 1 and row[4] == 1 else 0 for row in train]).reshape(-1, 1) # labels (y)

    test = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
    ])

    nn = NeuralNetwork(n_inputs=5, n_hidden=4, n_outputs=1, learning_rate=0.1)
    # With only 2 hidden neurons in one hidden layer, the network might underfit slightly on some cases, so having 4 hidden neurons,
    # generally improves convergence without overfitting for such a simple problem.
    nn.train(train, labels)

    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = nn.get_parameters()
    print("Learned weights from input layer to hidden layer: \n", weights_input_hidden)
    print("Learned biases of hidden layer: \n", bias_hidden)
    print("Learned weights from hidden layer to output layer : \n", weights_hidden_output)
    print("Learned biases of output layer: \n", bias_output)

    print("Predictions:\n")
    for t in test:
        y_pred = nn.predict([t])[0][0] #
        print(f"yes with conf={y_pred:.2f}" if (y_pred >= 0.5) else f"no with conf={1 - y_pred:.2f}")
