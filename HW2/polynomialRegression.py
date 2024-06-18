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
    # print("X_shrinked: ", X_shrinked.shape)
    # print("dependency_matrix: ", dependency_matrix.shape)
    # print("X_copy: ", X_copy.shape)
    return X_shrinked

class gradientDescent():

    def __init__(self, x, y, w, lr, num_iters, degree = 1, feat_dim = 1, dep_mat = None):
        self.x = x
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        self.w = w.copy()
        self.weight_history = [self.w]
        self.degree = degree
        self.feat_dim = feat_dim
        self.dep_mat = dep_mat
        self.vandermonde = shrink_vandermonde_matrix(self.x, self.degree, self.feat_dim, self.dep_mat)
        self.cost_history = [np.sum(np.square(self.predict(self.vandermonde)-self.y))/self.x.shape[0]]

    
    
    def gradient(self):
        gradient = np.zeros_like(self.w)
        
        ##############################################################################
        # TODO: Calculate gradient of gradient descent algorithm.                    #
        #                                                                            #
        ##############################################################################
        #  Replace "pass" statement with your code
        
        X_copy = self.vandermonde.copy()
        w_copy = self.w.copy()
        # print("X_copy: ", X_copy.shape)
        # print("w_copy: ", w_copy.shape)
        # print("y: ", self.y.shape)
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

    def cross_validate(self, k=5):
        """
        Performs k-fold cross-validation and returns the cross-validated error.

        Args:
        - k (int): Number of folds for cross-validation.

        Returns:
        - float: Cross-validated error.
        """
        np.random.seed(42)
        m = self._x_train.shape[1]
        fold_size = m // k
        indices = np.arange(m)
        np.random.shuffle(indices)

        cross_val_error = []

        ##############################################################################
        # TODO: The cross_validate function is designed to perform k-fold            #
        # cross-validation, a technique used to evaluate the performance of a        #
        #machine learning model. It takes the number of folds, k, as input and       #
        #splits the dataset into k equal-sized subsets. For each fold, one subset is #
        #used as the validation set, and the remaining k-1 subsets are used as the   #
        #training set. The function iterates through each fold, training a linear    #
        #regression model on the training data and evaluating its performance on     #
        #the validation data. The error of each fold is recorded, and the function   #
        #returns the average cross-validated error, providing a reliable             #
        #estimate of the model's generalization performance.                         #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        pass
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ############################################################################## 

        avg_cross_val_error = np.mean(cross_val_error)
        
        return avg_cross_val_error
        
class SGD():
    def __init__(self, x, y, w, lr, num_iters, degree = 1, feat_dim = 1, dep_mat = None):
        self.x = x
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        self.w = w.copy()
        self.weight_history = [self.w]
        self.degree = degree
        self.feat_dim = feat_dim
        self.dep_mat = dep_mat
        self.vandermonde = shrink_vandermonde_matrix(self.x, self.degree, self.feat_dim, self.dep_mat)
        self.cost_history = [np.sum(np.square(self.predict(self.vandermonde)-self.y))/self.x.shape[0]]
        
    def gradient(self, i):
        gradient = np.zeros_like(self.w)
        ##############################################################################
        # TODO: Calculate gradient of gradient descent algorithm.                    #
        #                                                                            #
        ##############################################################################
        #  Replace "pass" statement with your code
        
        x_slice = self.vandermonde[i].reshape(1, -1)
        gradient = (x_slice.T @ (x_slice @ self.w - self.y[i])) 
        
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
        # Steps of stochastic gradient descent algorithm:                            #
        #   1. Pick a sample from the training set.                                  #
        #   2. Calculate gradient of cost function. (Call gradient() function)       #
        #   3. Update w with gradient.                                               #
        #   4. Repeat 1-3 for all samples in the training set.                       #
        #   5. Log weight and cost for plotting in weight_history and cost_history.  #
        #   6. Repeat 1-5 until the cost change converges to epsilon or achieves     #
        # n_iterations.                                                              #
        # !!WARNING-1: Use Mean Square Error between predicted and  actual y values  #
        # for cost calculation.                                                      #
        #                                                                            #
        # !!WARNING-2: Be careful about lr can takes "diminishing". It will be used  #
        # for further test in the jupyter-notebook. If it takes lr decrease with     #
        # iteration as (lr =(1/k+1), here k is iteration).                           #
        ##############################################################################

        # Replace "pass" statement with your code
        X_copy = self.x.copy()
        
        while k < n_iterations:
            
            while not X_copy.shape[0] == 0:
                idx = np.random.randint(0, X_copy.shape[0])
                x = X_copy[idx]
                X_copy = np.delete(X_copy, idx, axis=0)
                
                if lr == "diminishing":
                    self.lr = 1 / (k + 1)
                
                gradient = self.gradient(idx)
                self.w = self.w - self.lr * gradient
                
            self.weight_history.append(self.w)
            cost = np.sum(np.square(self.predict(self.x)-self.y))/self.x.shape[0]
            
            self.cost_history.append(cost)
                
            if cost < self.epsilon:
                break
            
            k += 1
            
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return self.w, k
    
    def predict(self, x):
        if (x.shape[1] != self.vandermonde.shape[1]):
            x = np.c_[np.ones((x.shape[0], 1)), x]
        
        y_pred = np.zeros_like(self.y)

        y_pred = x.dot(self.w)

        return y_pred
        
    def cross_validate(self, k=5):
        """
        Performs k-fold cross-validation and returns the cross-validated error.

        Args:
        - k (int): Number of folds for cross-validation.

        Returns:
        - float: Cross-validated error.
        """
        np.random.seed(42)
        m = self._x_train.shape[1]
        fold_size = m // k
        indices = np.arange(m)
        np.random.shuffle(indices)

        cross_val_error = []

        ##############################################################################
        # TODO: The cross_validate function is designed to perform k-fold            #
        # cross-validation, a technique used to evaluate the performance of a        #
        #machine learning model. It takes the number of folds, k, as input and       #
        #splits the dataset into k equal-sized subsets. For each fold, one subset is #
        #used as the validation set, and the remaining k-1 subsets are used as the   #
        #training set. The function iterates through each fold, training a linear    #
        #regression model on the training data and evaluating its performance on     #
        #the validation data. The error of each fold is recorded, and the function   #
        #returns the average cross-validated error, providing a reliable             #
        #estimate of the model's generalization performance.                         #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        pass
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ############################################################################## 

        avg_cross_val_error = np.mean(cross_val_error)
        
        return avg_cross_val_error       
