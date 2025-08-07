# now building the logistic regression model using numpy
import numpy as np
import copy
from tqdm import tqdm

def timeit(func):
    """Decorator to time the execution of a function."""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))
    
    def cost_function(self, y_true, y_pred):
        m = y_true.shape[0]
        cost = -(1/m) *np.sum (-y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred))
        return cost
    
    def compute_gradient(self, X, y):
        # X is feature matrix mXn
        # y_pred is predicted values mX1
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        z = np.array(z, dtype=float)
        f_wb = self.sigmoid(z)
        # compute gradients
        dj_dw = (1/m) * np.dot((f_wb - y), X)
        dj_db = (1/m) * np.sum(f_wb - y)
        
        return dj_dw, dj_db
    # time taken for one iteration is O(m*n) where m is number of samples and n is number of features
    
    def gradient_descent(self, X, y, learning_rate=0.01, num_iterations=100):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        J_history = []
        w = copy.deepcopy(self.weights)
        b = copy.deepcopy(self.bias)
        print("Shape of X:", X.shape)
        print("Shape of y:", y.shape)
        print("Initial weights:", w)
        print("Initial bias:", b)
        for i in tqdm(range(num_iterations)):
            z = np.dot(X, w) + b
            y_pred = self.sigmoid(z)
            cost = self.cost_function(y, y_pred)
            J_history.append(cost)
            dj_dw, dj_db = self.compute_gradient(X, y)
            w -= learning_rate * dj_dw
            b -= learning_rate * dj_db
        self.weights = w
        self.bias = b
        return J_history
    
    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            y_pred = self.sigmoid(np.dot(X[i], self.weights) + self.bias)
            y_pred = 1 if y_pred >= 0.5 else 0
            preds.append(y_pred)
        return preds
    
    def fit(self, X, y, learning_rate=0.01, num_iterations=10):
        J_history = self.gradient_descent(X, y, learning_rate, num_iterations)
        return J_history