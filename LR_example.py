import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionUsingGD:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    eta : float
        Learning rate
    n_iterations : int
        No of passes over the training set
    Attributes
    ----------
    w_ : weights/ after fitting the model
    bias_ : bias / after fitting the model
    cost_ : total error of the model after each iteration
    """

    def __init__(self, eta=0.05, n_iterations=10000):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """

        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        self.bias_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_) + self.bias_
            residuals = y_pred - y
            gradient_vector_weight = np.dot(x.T, residuals)
            gradient_vector_bias = np.sum(residuals) 
            self.w_ -= (self.eta / m) * gradient_vector_weight
            self.bias_ -= (self.eta / m) * gradient_vector_bias
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self

    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(x, self.w_) + self.bias_
    
    
def run():
    # generate random data-set
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.rand(100, 1)
    
    # Model Initialization
    model = LinearRegressionUsingGD()
    # Fit the data (train the model)
    model.fit(x, y)
    #model = LinearRegression().fit(x, y)
    
    # Predict
    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')
    
    
    # model evaluation
    rmse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    
    # plot
    plt.scatter(x,y,s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y_pred, color='r')
    plt.show()
    
    
if __name__ == '__main__':
    run()
