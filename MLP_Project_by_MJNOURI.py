import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plot_writer_variable_x = []
plot_writer_variable_y = []

# Set working directory and load data
iris = pd.read_csv(r'F:\Projects\ANN project\iris\iris.data')
iris.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'class']

# Create numeric classes for species (0,1,2);
iris.loc[iris['class'] == 'Iris-virginica', 'class'] = 0
iris.loc[iris['class'] == 'Iris-versicolor', 'class'] = 1
iris.loc[iris['class'] == 'Iris-setosa', 'class'] = 2
iris = iris[iris['class'] != 2]
print(iris)
# iris.loc[iris['Name'] == 'Iris-setosa'] = 2

# Create Input and Output columns
X = iris[['Petal Length', 'Petal Width']].values.T
Y = iris[['class']].values.T
Y = Y.astype('uint8')


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)  # we set up a seed so that our output matches ours although the initialization is random.

    W1 = np.random.randn(n_h, n_x) * 0.01  # weight matrix of shape (n_h, n_x)
    b1 = np.zeros(shape=(n_h, 1))  # bias vector of shape (n_h, 1)
    W2 = np.random.randn(n_y, n_h) * 0.01  # weight matrix of shape (n_y, n_h)
    b2 = np.zeros(shape=(n_y, 1))  # bias vector of shape (n_y, 1)

    # store parameters into a dictionary
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# Function to define the size of the layer
def layer_sizes(X, Y):
    n_x = X.shape[0]  # size of input layer
    n_h = 6  # size of hidden layer
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)


def forward_propagation(X, parameters):
    # retrieve intialized parameters from dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Implement Forward Propagation to calculate A2 (probability)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)  # tanh activation function
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # sigmoid activation function

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    m = Y.shape[1]  # number of training examples

    # Retrieve W1 and W2 from parameters
    W1 = parameters['W1']
    W2 = parameters['W2']

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m

    return cost


def backward_propagation(parameters, cache, X, Y):
    # Number of training examples
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']
    ### END CODE HERE ###

    # Retrieve A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        #Plot Writer Variable
        plot_writer_variable_x.append(i)
        plot_writer_variable_y.append(cost)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters, n_h


parameters = nn_model(X, Y, n_h=6, num_iterations=10000, print_cost=True)

plt.plot(plot_writer_variable_x, plot_writer_variable_y)
plt.title("Cost of each iteration ")
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
