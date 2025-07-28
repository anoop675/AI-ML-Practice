"""
Implementation of deep neural network (with single hidden layer), to make it capable of identifying any 5 bits with first and last bits as 1
>>> NOTE: This ∂L/∂ŷ should be read like (Loss gradient w.r.t output from output layer). Likewise for others

>>> NOTE: Convergence generally refers to the point where the model's performance (often measured by loss) stops improving significantly or reaches a stable state during training

>>> NOTE: Xavier initialization, also known as Glorot initialization, is used to initialize the weights in a neural network to prevent vanishing or exploding gradients, 
especially in deep networks with many layers. It aims to maintain a similar variance of activations across layers during both forward and backward propagation, which helps stabilize training. 
1. The Problem: Vanishing/Exploding Gradients
In deep neural networks, during backpropagation, gradients can become very small (vanishing) or very large (exploding) as they are passed through multiple layers.
This can significantly slow down or even prevent learning, as the updates to the weights become negligible or too drastic. 
(To know if the gradients are updating very small (vanishing) or very large (exploding), check the measure of the total gradient norm after every update (epoch)). 
This problem is often worsened by certain activation functions, like sigmoid and tanh, which can compress the signal. 
2. Xavier Initialization's Solution
Xavier initialization addresses this by carefully scaling the weights based on the number of input and output connections to each layer. 
The goal is to keep the variance of the activations (and thus the gradients) roughly the same across each layer. 
This ensures that the signal doesn't shrink or blow up as it propagates through the network. 
3. How it Works
Xavier initialization typically draws the initial weights from a normal distribution with a mean of 0 and a standard deviation that depends on the number of input and output units of a layer. 
The formula for the standard deviation often involves the square root of 2 divided by the sum of the number of inputs and outputs to the layer. 
There are also uniform distributions used, where the weights are drawn from a range based on the number of inputs and outputs. 
4. Why it's important for sigmoid/tanh
Xavier initialization is particularly helpful when using activation functions like sigmoid and tanh, as they can be sensitive to the scale of the inputs. 
By keeping the variance of activations relatively constant, Xavier initialization helps these activation functions operate in their more sensitive regions, where they can effectively learn. 

>>> Types of Training modes (for this implementation we are using Batch Gradient Descent)
| Type                       | Description                                    | Pros                            | Cons                          |
| -------------------------- | ---------------------------------------------- | ------------------------------- | ----------------------------- |
| **Batch Gradient Descent** | Train on **entire dataset** in each epoch      | Stable gradients, deterministic | Slow for large datasets       |
| **Stochastic GD**          | Train on **one sample at a time**              | Fast updates, more noise        | Less stable, noisy loss       |
| **Mini-Batch GD**          | Train on **small chunks** of the data (common) | Best of both worlds             | Slightly complex to implement |

Vanishing Gradient problem vs Exploding Gradients problem
| **Aspect**             | **Vanishing Gradients**                                                              | **Exploding Gradients**                                                        |
| -----------------------| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **Definition**         | Gradients shrink to near-zero during backpropagation                                 | Gradients grow excessively large during backpropagation                        |
| **Gradient Magnitude** | Extremely small (e.g., < 10^{-4})                                                    | Extremely large (e.g., > 10^2$, $10^3$)                                        |
| **Effect on Learning** | Learning becomes very slow or stalls in earlier layers                               | Causes instability, large weight updates, or NaNs                              |
| **Loss Behavior**      | Loss plateaus (no improvement across epochs)                                         | Loss may oscillate wildly or diverge                                           |
| **Gradient Norms**     | Norms approach zero                                                                  | Norms blow up exponentially                                                    |
| **Symptoms**           | No weight updates, Model predicts same value, Low accuracy                           | Model output is NaN or inf, Loss becomes NaN<, Unstable training               |
| **Where it happens**   | Mostly in early layers of deep networks                                              | In deeper or all layers, depending on architecture                             |
| **Common Causes**      | Sigmoid/tanh activations, Poor initialization, Too small LR Deep nets                | Poor weight initialization, High learning rate, No gradient clipping           |
| **Solutions**          | ReLU or Leaky ReLU, Xavier/He init, BatchNorm, Residual connections                  | Gradient clipping, Normalize inputs, Smaller learning rate                     |
| **Typical Domains**    | Deep feedforward nets, RNNs (esp. LSTMs, GRUs)                                       | Deep RNNs, CNNs, or deep MLPs with poor scaling                                |

NOTE:
    If A and B are matrices or column vectors (matrices with single column)
    where A.shape = (n, k) and B.shape = (k, m),
        A @ B (or) np.dot(A, B) (or) A.dot(B) performs matrix multiplication (provided the inner dimensions (k) are the same)

    If A and B are matrices or column vectors
    where A.shape = B.shape = (a, b)
        A * B performs element-wise multiplication
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score

# Activation function used is sigmoid function
def sigmoid(x): # performs sigmoid function on all values at once in vector
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): # performs derivative on all values at once (outputs of sigmoid functions) in vector
    return x * (1 - x)

# Loss (Cost) function used is MSE (Mean Squared Error) => sum[(y - ŷ)^2] / n
def mean_squared_error(y_actual, y_predicted): # y_actual and y_predicted are vectors
    error = y_actual - y_predicted
    mse = np.mean(error ** 2)
    #print(mse.shape) # scalar value
    return mse # MSE is a scalar value here

def binary_cross_entropy(y_true, y_pred, eps=1e-15): # y_true and y_pred are vectors
    y_pred = np.clip(y_pred, eps, 1 - eps)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    #print(bce.shape) # scalar value
    return bce

# derivative of MSE is -2(y - ŷ) NOTE: typically we drop the constant -2 (since it's absorbed by the learning rate in gradient descent algorithm)
def mean_squared_error_derivative(y_actual, y_predicted): # y_actual and y_predicted are vectors
    return y_actual - y_predicted # return a vector (or an array of the same shape as y_true and y_pred) of gradients, where each element corresponds to the ∂L/∂ŷ for each individual ŷ value

def binary_cross_entropy_derivative(y_true, y_pred, eps=1e-15): # y_true and y_pred are vectors
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))  # return a vector (or an array of the same shape as y_true and y_pred) of gradients, where each element corresponds to the ∂L/∂ŷ for each individual ŷ value


class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.1):
        self.d = n_inputs
        self.n = n_hidden
        self.m = n_outputs
        self.learning_rate = learning_rate

        # Initializing weights between each layer
        self.W_input_hidden = np.random.randn(self.d, self.n) * 0.1
        # Use Xavier (Glorot) Weight Initilization for better scaling in deep neural networks
        #limit_for_W_input_hidden = np.sqrt(6 / (self.d + self.n)) #Xavier initilization formula (for sigmoid/tanh)
        #self.W_input_hidden = np.random.uniform(-limit_for_W_input_hidden, limit_for_W_input_hidden, (self.d, self.n))

        
        self.W_hidden_output = np.random.randn(self.n, self.m) * 0.1
        # Use Xavier (Glorot) Weight Initilization for better scaling in deep neural networks
        #limit_for_W_hidden_output = np.sqrt(6 / (self.n + self.m)) #Xavier initilization formula (for sigmoid/tanh)
        #self.W_hidden_output = np.random.uniform(-limit_for_W_hidden_output, limit_for_W_hidden_output, (self.n, self.m))

        # Initializing biases for each layer
        self.b_hidden = np.zeros((1, self.n))
        self.b_output = np.zeros((1, self.m))

    def get_parameters(self):
        return [self.W_input_hidden, self.W_hidden_output, self.b_hidden, self.b_output]

    def forward_propagation(self, X):
        # input layer to hidden layer
        self.z_hidden = X @ self.W_input_hidden + self.b_hidden # computes z_hidden = W_input_hidden * X + b_hidden
        self.a_hidden = sigmoid(self.z_hidden) # computes a_hidden = sigmoid(z_hidden)

        # hidden layer to output layer
        self.z_out = self.a_hidden @ self.W_hidden_output + self.b_output # computes z_out = W_hidden_out * a_hidden + b_out
        self.a_out = sigmoid(self.z_out) # computes a_out = sigmoid(z_out)

        return self.a_out # ŷ = a_out

    def backward_propagation(self, X, y):
        # getting the loss for calculating its gradient wrt each parameter (as done by mean_squared_error_derivative())
        # and for evaluating the loss of the network after each epoch
        loss = binary_cross_entropy(y, self.a_out) # MSE is causing vanishing gradients problem (due to sigmoid saturation), so using BCE loss for classification problems

        #--------------Chain Rule part---------------
        # Gradients for output layer
        # ∂L/∂W_hidden_output = (∂L/∂ŷ) * (∂ŷ/∂z_out) * (∂z_out/∂W_hidden_output)
        # ∂L/∂b_output = (∂L/∂ŷ) * (∂ŷ/∂z_out) * (∂z_out/∂b_output)

        dloss_da_out = binary_cross_entropy_derivative(y, self.a_out) # computes ∂L/∂ŷ for output layer NOTE: ŷ is nothing but a_out
        da_out_dz_out = sigmoid_derivative(self.a_out) # computes ∂ŷ/∂z_out for output layer
        dloss_dz_out = dloss_da_out * da_out_dz_out # computes ∂L/∂z_out = (∂L/∂ŷ) * (∂ŷ/∂z_out) for output layer, NOTE: element wise multiplication of two matrices (column vectors) are done here, not matrix multiplication

        self.dloss_dW_hidden_output = self.a_hidden.T @ dloss_dz_out # computes ∂L/∂W_hidden_output = (∂L/∂z_out) * (∂z_out/∂W_hidden_output)
        self.dloss_db_output = np.sum(dloss_dz_out, axis=0, keepdims=True) # computes ∂L/∂b_output
        '''
        NOTE : 
            In line 116, Why is a_hidden.T used instead of ∂z_out/∂W_hidden_output ?
            
            Solution:
                We know according to chain rule, 
                >>> ∂L/∂W_hidden_output = (∂L/∂z_out) * (∂z_out/∂W_hidden_output) ---> (1),
                
                To find: ∂z_out/∂W_hidden_output from (1),
                
                We know, 
                >>> z_out = a_hidden @ W_hidden_output + b_output (considering the equivalence of inner dimensions of a_hidden and W_hidden_output for dot product) 
                Differentiating wrt W_hidden_output on both sides, we get,
                >>> ∂z_out/∂W_hidden_output = ∂(a_hidden @ W_hidden_output)/∂W_hidden_output + ∂b_output/∂W_hidden_output
                >>> ∂z_out/∂W_hidden_output = a_hidden^T + 0 
                >>> ∂z_out/∂W_hidden_output = a_hidden^T ---> (2)    
                since,
                    ∂(A @ B)/∂B = A^T in vector calculus, provided A is a vector and B is a matrix and @ means dot product,
                    ∂(X)/∂Y leads to constant in this cause a null vector 0, provided A is a vector and B is a matrix 
                
                Substituting (2) in (1),
                >>> ∂L/∂W_hidden_output = a_hidden^T @ (∂L/∂z_out) (considering the equivalence of inner dimensions of a_hidden^T and ∂L/∂z_out for dot product) 
        
            In line 117, Why is ∂L/∂b_output = np.sum(∂L/∂z_out, axis=0, keepdims=True) ?
            
            Solution:
                We know according to chain rule,
                >>> ∂L/∂b_output = (∂L/∂z_out) * (∂z_out/∂b_output) ---> (1),
                
                To find: ∂z_out/∂b_output from (1)
                
                We know z_out = a_hidden @ W_hidden_output + b_output (considering the equivalence of inner dimensions of a_hidden and W_hidden_output for dot product) 
                Differentiating wrt b_output on both sides, we get,
                >>> ∂z_out/∂b_output = ∂(a_hidden @ W_hidden_output)/∂b_output + ∂b_output/∂b_output
                >>> ∂z_out/∂b_output = 0 + 1
                >>> ∂z_out/∂b_output = 1 ---> (2)
                 since,
                    ∂(A)/∂B = 0 (null vector) , provided A is a vector and B is a vector,
                    ∂(B)/∂B = 1 (ones vector) , provided A is a vector and B is a vector
                
                Substituting (2) in (1),
                >>> ∂L/∂b_output = ∂L/∂z_out
                Because dloss_dz_out contains per-sample gradients. But the bias b_output which is one bias vector is shared across all samples in the batch (broadcasted).
                So to compute the gradient of the total loss w.r.t. the bias (∂L/∂b_output), we need to aggregate the individual sample gradients (∂L/∂z_out).
                Hence,
                >>> ∂L/∂b_output = sum( ∂L/∂z_out (i) ) where m is the number of rows
                                  i to m
        '''

        # Error to propagate to hidden layer
        # ∂L/∂W_input_hidden = (∂L/∂ŷ) * (∂ŷ/∂z_out) * (∂z_out/∂a_hidden) * (∂a_hidden/∂z_hidden) * (∂z_hidden/∂W_input_hidden)
        # ∂L/∂b_hidden = (∂L/∂ŷ) * (∂ŷ/∂z_out) * (∂z_out/∂a_hidden) * (∂a_hidden/∂z_hidden) * (∂z_hidden/∂b_hidden)

        dloss_da_hidden = dloss_dz_out @ self.W_hidden_output.T # computes ∂L/∂a_hidden = (∂L/∂z_out) * (∂z_out/∂a_hidden) for hidden layer. Instead of computing ∂z_out/∂a_hidden, we used (W_hidden_output)^T
        da_hidden_dz_hidden = sigmoid_derivative(self.a_hidden) # computes ∂a_hidden/∂z_hidden for hidden layer
        dloss_dz_hidden = dloss_da_hidden * da_hidden_dz_hidden # computes ∂L/∂z_hidden = (∂L/∂a_hidden) * (∂a_hidden/∂z_hidden) for hidden layer, NOTE: element wise multiplication of two matrices (column vectors) are done here, not matrix multiplication

        self.dloss_dW_input_hidden = X.T @ dloss_dz_hidden  # computes ∂L/∂W_input_hidden = (∂L/∂z_hidden) * (∂z_hidden/∂W_input_hidden)
        self.dloss_db_hidden = np.sum(dloss_dz_hidden, axis=0, keepdims=True) # computes ∂L/∂b_hidden
        '''
        NOTE : 
            In line 171, Why is W_hidden_output.T used instead of ∂z_out/∂a_hidden ?
            Solution: same case for line 116
            
            In line 175, Why is X.T used instead of ∂z_hidden/∂W_input_hidden ?
            Solution: same case for line 116
            
            In line 176, ∂loss/∂b_hidden = np.sum(dloss_dz_hidden, axis=0, keepdims=True) ?
            Solution same case for line 117
        '''
        #--------------------------------------------

        #-----------Gradient Descent part------------
        # Update weights and biases
        self.W_input_hidden -= self.learning_rate * self.dloss_dW_input_hidden # computes W_input_hidden = W_input_hidden - learning_rate * (∂L/∂W_input_hidden)
        self.b_hidden -= self.learning_rate * self.dloss_db_hidden # computes b_hidden = b_hidden - learning_rate * (∂L/∂b_hidden)

        self.W_hidden_output -= self.learning_rate * self.dloss_dW_hidden_output # computes W_hidden_output = W_hidden_output - learning_rate * (∂L/∂W_hidden_output)
        self.b_output -= self.learning_rate * self.dloss_db_output # computes b_output = b_output - learning_rate * (∂L/∂b_output)
        #--------------------------------------------
        # Return mean squared error for monitoring
        return loss

    def train(self, X_train, X_val, y_train, y_val, epochs=10000): # Using Batch Training as the training mode for this problem
        for epoch in range(epochs):
            #self.forward_propagation(X)
            #loss = self.backward_propagation(X, y)

            # Forward pass and backward pass on training set
            self.forward_propagation(X_train)
            train_loss = self.backward_propagation(X_train, y_train)

            # model performance evaluation on training set
            train_preds= (self.a_out > 0.5).astype(int)
            train_precision = precision_score(y_train, train_preds, zero_division=1)
            train_f1 = f1_score(y_train, train_preds, zero_division=1)

            # Forward pass on validation set
            val_output = self.forward_propagation(X_val)
            val_loss = binary_cross_entropy(y_val, val_output)

            # model performance evaluation on validation set
            val_preds = (val_output > 0.5).astype(int)
            val_precision = precision_score(y_val, val_preds, zero_division=1)
            val_f1 = f1_score(y_val, val_preds, zero_division=1)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}:")
                print(f"  Train Loss: {train_loss:.4f}   Val Loss: {val_loss:.4f}")
                print(f"  Train Precision: {train_precision:.4f}   Val Precision: {val_precision:.4f}")
                print(f"  Train F1 Score: {train_f1:.4f}   Val F1: {val_f1:.4f}")
                print(f"  Gradient Norm: {self.get_gradient_norms():.4f}")
                # If total_norm becomes very large (e.g. thousands or millions), that's a sign of exploding gradients.

        return self.get_parameters() # for seeing the learnt (optimized) (fine-tuned) weights and biases

    def predict(self, X):
        output = self.forward_propagation(X)
        return output

    # returns L2 norm (Euclidean norm) of all the model's gradients combined, to know how large the gradient updates are at the current step.
    # Lp norm = ||x||p = (sum(|x|^p)^(1/p). For a Euclidean norm, p = 2 (L2 norm). NOTE: x is a vector of gradient
    '''
    The gradient norm (typically the L2 norm) reflects the overall magnitude of the parameter updates.
    If:
        Gradient norm is very close to 0, then gradients are vanishing, which means very little learning.
        Gradient norm tends to infinity, then gradients are explodingm which means unstable learning.
        Gradient norm in a moderate range (e.g., 0.01 – 10), means training is likely stable.

    If you're training a neural network and plot the gradient norm per epoch:
        A flat line near zero, means vanishing gradients, means your weights aren't updating much.
        A line that shoots up exponentially, means exploding gradients, means your weights are blowing up.
        A smooth, slowly decreasing line, means good training dynamics.
    '''
    def get_gradient_norms(self):
        list_of_gradients = [
            self.dloss_dW_input_hidden,
            self.dloss_db_hidden,
            self.dloss_dW_hidden_output,
            self.dloss_db_output
        ]
        total_norm = 0
        for gradients in list_of_gradients:
            param_norm = np.linalg.norm(gradients)
            total_norm += param_norm ** 2
        return np.sqrt(total_norm)

if __name__ == "__main__":

    np.random.seed(42)

    X = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]
    ])

    # Condition: first bit AND last bit are both 1
    y = np.array([(x[0] == 1 and x[-1] == 1) for x in X], dtype=int)

    nn = NeuralNetwork(n_inputs=5, n_hidden=4, n_outputs=1, learning_rate=0.01)
    # With only 2 hidden neurons in one hidden layer, the network might underfit slightly on some cases, so having 4 hidden neurons,
    # generally improves convergence without overfitting for such a simple problem.

    # Splitting off 10% for test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
        #X, y, test_size=0.10, random_state=42
    )

    # Splitting remaining 90% into 80% train and 20% validation (which gives 72% train, 18% validation overall)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=0.20, random_state=42
        #X_temp, y_temp, test_size=0.20, random_state=42
    )

    #nn.train(train, labels)
    nn.train(X_train, X_val, y_train.reshape(-1, 1), y_val.reshape(-1, 1), epochs=10000)

    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = nn.get_parameters()
    print("Learned weights from input layer to hidden layer: \n", weights_input_hidden)
    print("Learned biases of hidden layer: \n", bias_hidden)
    print("Learned weights from hidden layer to output layer : \n", weights_hidden_output)
    print("Learned biases of output layer: \n", bias_output)

    print("Predictions:\n")
    for t in X_test:
        y_pred = nn.predict([t])[0][0]
        print(f"Input: {t} -> ")
        print(f"yes with conf={y_pred:.2f}" if (y_pred >= 0.5) else f"no with conf={1 - y_pred:.2f}")
    '''
        NOTE: since this problem is very simple, the model overfits. Because the pattern is easy and deterministic, which makes
        the model memorize it instead of generalizing, because the problem is simple, deterministic, data is less in size and
        contains less noise. 

        How is it overfitting?
            Because Training loss ≈ 0
            Because Validation loss ≈ 0
            Perfect F1 and precision (all close to 1)
            100 % confidence predictions
            
        Hence the model overfitting. This is okay for simple problems with simple datasets.
    '''
