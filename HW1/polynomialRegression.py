import numpy as np

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def shrink_vandermonde_matrix(X, degree, feat_dim, dep_mat):
    # w_dim = X.shape[1] + 1 # for the bias term
    w_dim = degree + 1


    X_copy = np.c_[np.ones((X.shape[0], 1)), X]

    for i in range(2, degree + 1): # create the Vandermonde matrix
        X_copy = np.c_[X_copy, X**i]
        
    if dep_mat is None:
        dependency_matrix = np.zeros((X_copy.shape[1], w_dim))
        dependency_matrix[0, 0] = 1
        w_i = 1
        # print(dependency_matrix.shape)
        # print(X_copy.shape)
        for i in range(1, X_copy.shape[1], feat_dim): # shrink the Vandermonde matrix
            # print(i, i+feat_dim, w_i)
            dependency_matrix[i:i+feat_dim, w_i] = 1
            w_i += 1
    else:
        dependency_matrix = dep_mat
        
    # print("X_copy: ", X_copy.shape)
    # for i in range(1, X_copy.shape[1], feat_dim): # shrink the Vandermonde matrix
    #     X_shrinked = np.c_[X_shrinked, X_copy[:,i:i+feat_dim-1].sum(axis=1)[:, np.newaxis]]
        
    X_shrinked = X_copy @ dependency_matrix
    
    return X_shrinked

def leastSquares(X, Y, degree, feat_dim = 1, dep_mat = None):

    """
    Input:
    X and Y are two-dim numpy arrays.
    X dims: (N of samples, feature dims)
    Y dim: (N of samples, response dims)
    degree: Degree of the polynomial
    Output:
    Weight (Coefficient) vector
    Vandermonde Matrix
    """
    # Add a column of ones for the intercept
    if degree < 1:
        raise ValueError("Polynomial degree must be greater than or equal to 1.")

    # Initialize the Vandermonde matrix with the original features
    X_poly = X.copy()
        
    ##############################################################################
    # TODO: Implement least square theorem for polynomial regression.            #
    #                                                                            #
    # You may not use any built in function which directly calculate             #
    # Least squares except matrix operation in numpy.  			    #
    # You need to define and compute Vandermonde matrix first.                   #
    ##############################################################################
    # Replace "pass" statement with your code
    
    X_poly = shrink_vandermonde_matrix(X, degree, feat_dim, dep_mat)
    
    W = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ Y
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################

    return W, X_poly

class gradientDescent():

    def __init__(self, x, y, w, lr, num_iters, degree = 1, feat_dim = 1, dep_mat = None):
        self.x = x
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        
        # self.w = w.copy()
        # self.weight_history = [self.w]
        # self.cost_history = [np.sum(np.square(self.predict(np.c_[np.ones((self.x.shape[0], 1)), self.x])-self.y))/self.x.shape[0]]
        # self.degree = degree
        # self.feat_dim = feat_dim
        # self.dep_mat = dep_mat
        # self.vandermonde = shrink_vandermonde_matrix(self.x, self.degree, self.feat_dim, self.dep_mat)
        
        self.w = w.copy()
        self.weight_history = [self.w]
        self.degree = degree
        self.feat_dim = feat_dim
        self.dep_mat = dep_mat
        self.vandermonde = shrink_vandermonde_matrix(self.x, self.degree, self.feat_dim, self.dep_mat)
        self.cost_history = [np.sum(np.square(self.predict(self.vandermonde)-self.y))/self.x.shape[0]]
    
    def gradient(self): # TODO: gradient'te sikinti olabilir, LR cok cok kucuk olmak zorunda
        gradient = np.zeros_like(self.w)
        
        ##############################################################################
        # TODO: Calculate gradient of gradient descent algorithm.                    #
        #                                                                            #
        ##############################################################################
        #  Replace "pass" statement with your code
        
        # (V @ w - y_p) ^ 2  /  N = Mean Square Error
        # 2 * (V @ w - y_p) * V / N = Gradient_w  
        
        
        X_copy = self.vandermonde.copy()
        w_copy = self.w.copy()
        
        # X_copy = shrink_vandermonde_matrix(X_copy, self.degree, self.feat_dim)
        
        # for i in range(2, self.degree+1):
        #     X_copy = np.c_[X_copy, self.x**i] # 2 dimensional feature is added to the X_copy
            
        # ones_array = np.ones((X_copy.shape[0], 1))
        # X_copy = np.hstack((ones_array, X_copy)) # Vandermonde matrix
        
        # # Or we need to downsize the X_copy matrix so that we can use it in the matrix multiplication
        
        # last2_columns = np.sum(X_copy[:,-2:], axis=1)
        # second_last2_columns = np.sum(X_copy[:,-4:-2], axis=1)
        
        # # X_copy = np.delete(X_copy, [-2, -1], axis=1)
        # X_copy = np.c_[X_copy[:,0], second_last2_columns, last2_columns]
        
        # print(X_copy.shape, w_copy.shape, self.y.shape)
        gradient = (X_copy.T @ (X_copy @ w_copy - self.y)) / X_copy.shape[0] # 2 constant oldugu icin LR'in icinde kabul edilir
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
       
        return gradient
    
    def fit(self,lr=None, n_iterations=None):
        k = 0
        if n_iterations is None:
            n_iterations = self.num_iters
        if lr != None and lr != "diminishing":
            self.lr = lr
        
        ##############################################################################
        # TODO: Implement gradient descent algorithm.                                #
        #                                                                            #
        # You may not use any built in function which directly calculate             #
        # gradient.                                                                  #
        # Steps of gradient descent algorithm:                                       #
        #   1. Calculate gradient of cost function. (Call gradient() function)       #
        #   2. Update w with gradient.                                               #
        #   3. Log weight and cost for plotting in weight_history and cost_history.  #
        #   4. Repeat 1-3 until the cost change converges to epsilon or achieves     #
        # n_iterations.                                                              #
        # !!WARNING-1: Use Mean Square Error between predicted and  actual y values #
        # for cost calculation.                                                      #
        #                                                                            #
        # !!WARNING-2: Be careful about lr can takes "diminishing". It will be used  #
        # for further test in the jupyter-notebook. If it takes lr decrease with     #
        # iteration as (lr =(1/k+1), here k is iteration).                           #
        ##############################################################################

        # Replace "pass" statement with your code
        
        while k < n_iterations:
            
            if lr == "diminishing":
                self.lr = 1 / (k + 1)
            
            gradient = self.gradient()
            self.w = self.w - self.lr * gradient
            self.weight_history.append(self.w)
            # cost = np.sum(np.square(self.predict(np.c_[np.ones((self.x.shape[0], 1)), self.x])-self.y))/self.x.shape[0]
            cost = np.sum(np.square(self.predict(self.vandermonde)-self.y))/self.x.shape[0]
            
            # print("Cost: ", cost) # TODO: cost LR'in 0.01 gibi bir degerinde bile cok fazla diverge ediyor, optimizasyon lazim: feature scaling, data quality, poor initalization vb.
            
            self.cost_history.append(cost)
            
            if abs(cost) < self.epsilon:
                break
            
            k += 1
            
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return self.w, k
    
    def predict(self, x):
        
        y_pred = np.zeros_like(self.y)

        y_pred = x.dot(self.w)

        return y_pred

