import numpy as np

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class gradientDescent():
    def __init__(self, x, y, w, lr, num_iters):
        self.x = np.c_[np.ones((x.shape[0], 1)), x]
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        self.w = w.copy()
        self.weight_history = [self.w]
        self.cost_history = [np.sum(np.square(self.predict(self.x)-self.y))/self.x.shape[0]]
        
        # Cross-validation data
        self.cv_x_val = x
        self.cv_x_train = x
        self.cv_y_val = y
        self.cv_y_train = y
        
    def gradient(self, cv_flag=False):
        gradient = np.zeros_like(self.w)
        ##############################################################################
        # TODO: Calculate gradient of gradient descent algorithm.                    #
        #                                                                            #
        ##############################################################################
        #  Replace "pass" statement with your code

        if cv_flag:
            gradient = (self.cv_x_train.T @ (self.cv_x_train @ self.w - self.cv_y_train.reshape(-1, 1))) / self.cv_x_train.shape[0]
        else:
            gradient = (self.x.T @ (self.x @ self.w - self.y)) / self.x.shape[0] # 2 constant oldugu icin LR'in icinde kabul edilir
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return gradient
    
    
    def fit(self,lr=None, n_iterations=None, cv_flag=False):
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
        # !!WARNING-1: Use Mean Square Error between predicted and  actual y values  #
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
            
            gradient = self.gradient(cv_flag=cv_flag)

            self.w = self.w - self.lr * gradient
            self.weight_history.append(self.w)
            if cv_flag:
                cost = np.sum(np.square(self.predict(self.cv_x_val)-self.cv_y_val))/self.cv_x_val.shape[0]
            else:
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
        if (x.shape[1] != self.x.shape[1]):
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
        m = self.x.shape[0]
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
        
        for i in range(k):
            # Define the indices for the validation set
            val_indices = indices[i * fold_size: (i + 1) * fold_size]

            # print("val_indices: ", val_indices.shape)

            # Define the indices for the training set
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            # print("train_indices: ", train_indices.shape)

            # Split the data into validation and training sets
            self.cv_x_val, self.cv_y_val = self.x[val_indices], self.y[val_indices]
            # print("cv_x_val: ", self.cv_x_val.shape)
            # print("cv_y_val: ", self.cv_y_val.shape)
            
            self.cv_x_train, self.cv_y_train = self.x[train_indices], self.y[train_indices]
            # print("cv_x_train: ", self.cv_x_train.shape)
            # print("cv_y_train: ", self.cv_y_train.shape)

            # Train the model on the training data
            self.fit(cv_flag=True)


            
            # Evaluate the model on the validation data
            y_pred = self.predict(self.cv_x_val)
            
            # print("Fold: ", i)
            # print("Weights: ", self.w.shape)
            # print("y_pred: ", y_pred.shape)
            # print("cv_y_val: ", self.cv_y_val.shape)
            
            error = rel_error(self.cv_y_val, y_pred)
            cross_val_error.append(error)
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ############################################################################## 

        avg_cross_val_error = np.mean(cross_val_error)
        
        return avg_cross_val_error
        
class SGD():
    def __init__(self, x, y, w, lr, num_iters):
        self.x = np.c_[np.ones((x.shape[0], 1)), x]
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        self.w = w.copy()
        self.weight_history = [self.w]
        self.cost_history = [np.sum(np.square(self.predict(self.x)-self.y))/self.x.shape[0]]
        
        # Cross-validation data
        self.cv_x_val = x
        self.cv_x_train = x
        self.cv_y_val = y
        self.cv_y_train = y
        
    def gradient(self, i, cv_flag=False):
        gradient = np.zeros_like(self.w)
        ##############################################################################
        # TODO: Calculate gradient of gradient descent algorithm.                    #
        #                                                                            #
        ##############################################################################
        #  Replace "pass" statement with your code
        # print(self.x[i].shape, self.w.shape, self.y[i].shape)
        if cv_flag:
            x_slice = self.cv_x_train[i].reshape(1, -1)
            gradient = (x_slice.T @ (x_slice @ self.w - self.cv_y_train[i]))           
        else:
            x_slice = self.x[i].reshape(1, -1)
            gradient = (x_slice.T @ (x_slice @ self.w - self.y[i])) 
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        # print("Gradient: ", gradient.shape)
        return gradient
    
    
    def fit(self,lr=None, n_iterations=None, cv_flag=False):
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
        
        if cv_flag:
            X_copy = self.cv_x_train.copy()
        else:   
            X_copy = self.x.copy()
        
        while k < n_iterations:
            
            while not X_copy.shape[0] == 0:
                idx = np.random.randint(0, X_copy.shape[0])
                x = X_copy[idx]
                X_copy = np.delete(X_copy, idx, axis=0)
                
                if lr == "diminishing":
                    self.lr = 1 / (k + 1)
                
                gradient = self.gradient(idx, cv_flag=cv_flag)
                self.w = self.w - self.lr * gradient
                
            self.weight_history.append(self.w)
            if cv_flag:
                cost = np.sum(np.square(self.predict(self.cv_x_val)-self.cv_y_val))/self.cv_x_val.shape[0]
            else:
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
        if (x.shape[1] != self.x.shape[1]):
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
        m = self.x.shape[0]
        fold_size = m // k
        indices = np.arange(m)
        # print("Indices: ", indices.shape)
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
        
        for i in range(k):
            # Define the indices for the validation set
            val_indices = indices[i * fold_size: (i + 1) * fold_size]

            # print("val_indices: ", val_indices.shape)

            # Define the indices for the training set
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            # print("train_indices: ", train_indices.shape)

            # Split the data into validation and training sets
            self.cv_x_val, self.cv_y_val = self.x[val_indices], self.y[val_indices]
            # print("cv_x_val: ", self.cv_x_val.shape)
            # print("cv_y_val: ", self.cv_y_val.shape)
            
            self.cv_x_train, self.cv_y_train = self.x[train_indices], self.y[train_indices]
            # print("cv_x_train: ", self.cv_x_train.shape)
            # print("cv_y_train: ", self.cv_y_train.shape)

            # Train the model on the training data
            self.fit(cv_flag=True)
            
            # Evaluate the model on the validation data
            y_pred = self.predict(self.cv_x_val)
            
            # print("Fold: ", i)
            # print("Weights: ", self.w.shape)
            # print("y_pred: ", y_pred.shape)
            # print("cv_y_val: ", self.cv_y_val.shape)
            
            error = rel_error(self.cv_y_val, y_pred)
            cross_val_error.append(error)
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ############################################################################## 

        avg_cross_val_error = np.mean(cross_val_error)
        
        return avg_cross_val_error
