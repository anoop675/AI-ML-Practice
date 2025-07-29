"""
Implementation of a Multilayer Perceptron deep neural network to classify on whether a sequence of 5 bits has first bit and last bit as 1 or not.
Prerequisite: Understand the basic working of a single layer perceptron first to understand multilayer perceptron. Read "Singlelayer_Perceptron_ANN_from_scratch.ipynb"
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Use ReLU for hidden layers (instead of sigmoid), which avoids vanishing gradients

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig_x):
  return sig_x * (1 - sig_x)

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

class NeuralNetwork:
  def __init__(self, layer_sizes, learning_rate):
    self.layer_sizes = layer_sizes
    self.learning_rate = learning_rate

    self.list_of_weight_matrices = [] #list of weight matrices between the layers
    self.list_of_bias_vectors = [] #list of bias vectors for each layer

    #Initializing weights using Xavier weight initialization (as sigmoid activation is going to be used in network)
    for l in range(len(self.layer_sizes) - 1): # for 'l' layers, there will be 'l-1' weight matrices
      #Using Xavier weight initialization for each of the weights
      limit = np.sqrt(6 / (self.layer_sizes[l] + self.layer_sizes[l+1]))
      weight_matrix = np.random.uniform(-limit, limit, (self.layer_sizes[l], self.layer_sizes[l+1]))
      self.list_of_weight_matrices.append(weight_matrix)

    for l in range(len(self.layer_sizes) - 1): # except for the input layer, each layer will have a bias vector
      bias_vector = np.zeros((1, self.layer_sizes[l+1]))
      self.list_of_bias_vectors.append(bias_vector)

  def get_parameters(self):
    return self.list_of_weight_matrices, self.list_of_bias_vectors


  def forward_propagation(self, X):
    self.z_values = [] # z_values stores the pre-activation (intermediate) outputs from each layer
    self.a_values = [X] # a_values stores the activation outputs from each layer
    # NOTE: There are actually no activations used in the input layer, but inorder to maintain consistency (i.e. to avoid a_values[0] to be None),
    # we consider the activation outputs of the input layer (a_values[0]) as the input (X)
    a = X
    for W, b in zip(self.list_of_weight_matrices, self.list_of_bias_vectors):
      z = a @ W + b
      a = sigmoid(z)
      self.z_values.append(z)
      self.a_values.append(a)

    return a #the final output of network

  def backward_propagation(self, X, y):
    loss = binary_cross_entropy(y, self.a_values[-1])

    #--------------------Chain Rule part-------------------------------
    # computing gradient for output layer alone
    dloss_da_out = binary_cross_entropy_derivative(y, self.a_values[-1])
    da_out_dz_out = sigmoid_derivative(self.a_values[-1])
    dloss_dz_out = dloss_da_out * da_out_dz_out

    self.dloss_dW = [None] * (len(self.layer_sizes) - 1) # initializing a list of weight gradients of size (len(self.layer_sizes) - 1)
    self.dloss_db = [None] * (len(self.layer_sizes) - 1) # initializing a list of bias gradients of size (len(self.layer_sizes) - 1)

    for l in reversed(range(len(self.layer_sizes) - 1)): #start computing gradients from the second-last layer (hidden layer right before output layer)
        a_prev = self.a_values[l]
        self.dloss_dW[l] = a_prev.T @ dloss_dz_out
        self.dloss_db[l] = np.sum(dloss_dz_out, axis=0, keepdims=True)

        if l != 0:  # Don't compute dloss_dz_out for input layer
          dloss_da = dloss_dz_out @ self.list_of_weight_matrices[l].T
          da_dz = sigmoid_derivative(self.a_values[l])
          dloss_dz = dloss_da * da_dz
    #------------------------------------------------------------------

    #---------------------Gradient Descent part------------------------
    # Update weights and biases using gradient descent
    for l in range(len(self.layer_sizes) - 1):
        self.list_of_weight_matrices[l] -= self.learning_rate * self.dloss_dW[l]
        self.list_of_bias_vectors[l] -= self.learning_rate * self.dloss_db[l]
    #------------------------------------------------------------------

    # for evaluation at each pass
    return loss

  def total_gradient_norm(self):
    total_norm = 0
    for dw, db in zip(self.dloss_dW, self.dloss_db):
        total_norm += np.linalg.norm(dw)**2 + np.linalg.norm(db)**2
    return np.sqrt(total_norm)

  def train(self, X_train, X_val, y_train, y_val, epochs):
    for epoch in range(epochs):
      # Forward pass and backward pass on the training set
      a = self.forward_propagation(X_train)
      train_loss = self.backward_propagation(X_train, y_train)

      # performance on the training set
      train_pred = (a > 0.5).astype(int)
      train_accuracy = accuracy_score(y_train, train_pred)
      train_precision = precision_score(y_train, train_pred, zero_division=1)
      train_recall = recall_score(y_train, train_pred)
      train_f1 = f1_score(y_train, train_pred, zero_division=1)

      # Forward pass on the validation set
      val_output = self.forward_propagation(X_val)
      val_loss = binary_cross_entropy(y_val, val_output)

      # performance on the training set
      val_pred = (val_output > 0.5).astype(int)
      val_accuracy = accuracy_score(y_val, val_pred)
      val_precision = precision_score(y_val, val_pred, zero_division=1)
      val_recall = recall_score(y_val, val_pred)
      val_f1 = f1_score(y_val, val_pred, zero_division=1)


      if epoch % 1000 == 0:
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}   Val Loss: {val_loss:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.4f}   Val Accuracy: {val_accuracy:.4f}")
        print(f"  Train Precision: {train_precision:.4f}   Val Precision: {val_precision:.4f}")
        print(f"  Train Recall: {train_recall:.4f}   Val Recall: {val_recall:.4f}")
        print(f"  Train F1 Score: {train_f1:.4f}   Val F1: {val_f1:.4f}")
        print(f"  Total Gradient Norm: {self.total_gradient_norm():.4f}")

  def predict(self, X_test):
    test_pred = self.forward_propagation(X_test)
    return test_pred


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

  y = np.array([1 if row[0] == 1 and row[-1] == 1 else 0 for row in X])

  layer_sizes = [5, 4, 1] #E.g. [5,4,1] means 5 neurons in input layer, 4 neurons for single hidden layer, 1 neuron for output layer

  nn = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=0.1) # Considering a Single layer perceptron for now

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

  nn.train(X_train, X_val, y_train.reshape(-1, 1), y_val.reshape(-1, 1), epochs=10000)

  list_of_weight_matrices, list_of_bias_vectors = nn.get_parameters()
  for l in range(len(layer_sizes) - 1):
    print(f"Learned weights between layer #{l} to layer #{l+1}: \n", list_of_weight_matrices[l])
    print(f"Learned biases for layer #{l+1}: \n", list_of_bias_vectors[l])

  print("Predictions:\n")
  for t in X_test:
      y_pred = nn.predict([t])[0][0]
      print(f"Input: {t} -> ")
      print(f"yes with conf={y_pred:.2f}" if (y_pred >= 0.5) else f"no with conf={1 - y_pred:.2f}")
    
