import numpy as np
np.random.seed(42)
class LogisticRegression:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Constructor assumes an x_train matrix in which each column contains an instance.
        Vector y_train contains binary labels for each instance (0 or 1).

        Constructor initializes the weights W and B, alpha, and a one-vector Y containing the labels
        of the training set.
        """
        np.random.seed(42) # was 42
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._m = x_train.shape[1] # Number of instances
        
        self._W = np.random.randn(1, x_train.shape[0]) * 0.02
        self._B = 0
        self._Y = y_train.reshape(1, -1)  # Binary labels, no need for one-hot encoding
        self._alpha = 0.001 # learning rate 0.05

    def sigmoid(self, Z):
        """
        Computes the sigmoid value for all values in vector Z.
        """
        ##############################################################################
        # TODO: Compute the sigmoid value for all values in input vector             #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        sigmoid_val = 1 / (1 + np.exp(-Z))
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return sigmoid_val

    def h_theta(self, X):
        """
        Computes the value of the hypothesis according to the logistic regression rule.
        """
        ##############################################################################
        # TODO: Implement Logistic Regression on input X.                            #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        h_theta = self.sigmoid(self._W @ X + self._B) # check the shapes !!!
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return h_theta

    @staticmethod
    def confusionMatrix(y_true, y_pred):
        # Create a confusion matrix
        true_pos = np.sum((y_pred == 1) & (y_true == 1))
        true_neg = np.sum((y_pred == 0) & (y_true == 0))
        false_pos = np.sum((y_pred == 1) & (y_true == 0))
        false_neg = np.sum((y_pred == 0) & (y_true == 1))
        
        return true_pos, true_neg, false_pos, false_neg
    
    def train_binary_classification(self, iterations, metricMatrix = False):
        """
        Performs a number of iterations of gradient descent equal to the parameter passed as input.
        Returns a list with the percentage of instances classified correctly in the training and test sets.
        """
        classified_correctly_train_list = []
        classified_correctly_test_list = []
        ##############################################################################
        # TODO: Implement stochastic gradient descent algorithm on minimizing Mean Squared      #
        # Error cost. You need to obtain partial derivatives before coding up your   # 
        # solutions. This method returns correctly classified train and test samples # 
        # computed at each iteration (Or each N iterations.)                         #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        # 1/N sum (sigmoid(xT.w) - y) ^ 2
        
        k = 0
        weight_history = []
        cost_history = []
        train_metrics = []
        test_metrics = []

        # Stochastic Gradient Descent
        while k < iterations:
            X_copy = self._x_train.copy()
            
            while not X_copy.shape[1] == 0:
                idx = np.random.randint(0, X_copy.shape[1])
                x = X_copy[:, idx].reshape(-1, 1)
                X_copy = np.delete(X_copy, idx, axis=1)
                
                # Debug
                # print("X_copy shape: ", X_copy.shape)
                # print("X shape: ", x.shape)
                # print("W shape: ", self._W.shape)
                # print("Y shape: ", self._Y.shape)
                
                # Calculate the gradient
                # grad_w = (self.h_theta(x) - self._Y[idx]) @ x / self._m
                grad_w = float(self._Y[:, idx] - self.h_theta(x)) * float(self.h_theta(x)) * float(1 - self.h_theta(x)) * x.T / self._m
                
                grad_b = float(self._Y[:, idx] - self.h_theta(x)) * float(self.h_theta(x)) * float(1 - self.h_theta(x)) / self._m
            
                # print("grad_w: ", grad_w)
                # print("grad_b: ", grad_b)
            
                self._W -= self._alpha * grad_w
                self._B -= self._alpha * grad_b 
                
            weight_history.append(self._W)

            cost = np.sum(np.square(self.h_theta(self._x_train) - self._y_train)) / self._m
            
            cost_history.append(cost)
                
            if metricMatrix:
                # Calculate Metrics
                train_predictions = self.h_theta(self._x_train) >= .5 # >= or > ???
                test_predictions = self.h_theta(self._x_test) >= .5
                
                true_pos_train, true_neg_train, false_pos_train, false_neg_train = self.confusionMatrix(self._y_train, train_predictions)
                true_pos_test, true_neg_test, false_pos_test, false_neg_test = self.confusionMatrix(self._y_test, test_predictions)
                
                # Calculate Metrics for Stochastic Gradient Descent (SGD) with Mean Squared Error (MSE)
                train_accuracy = (true_pos_train + true_neg_train) / (true_pos_train + true_neg_train + false_pos_train + false_neg_train)
                test_accuracy = (true_pos_test + true_neg_test) / (true_pos_test + true_neg_test + false_pos_test + false_neg_test)

                train_precision = true_pos_train / (true_pos_train + false_pos_train)
                test_precision = true_pos_test / (true_pos_test + false_pos_test)

                train_recall = true_pos_train / (true_pos_train + false_neg_train)
                test_recall = true_pos_test / (true_pos_test + false_neg_test)

                train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall)
                test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

                train_metrics.append([train_accuracy, train_precision, train_recall, train_f1]) # Hold as 2D array
                test_metrics.append([test_accuracy, test_precision, test_recall, test_f1])
                
            else:
                # Calculate predictions for training set
                train_predictions = self.h_theta(self._x_train)
                correctly_classified_train = np.sum((train_predictions >= 0.5) == self._y_train)
                percent_correct_train = (correctly_classified_train / len(self._y_train)) * 100
                # print("Percent correct train: ", percent_correct_train)
                classified_correctly_train_list.append(percent_correct_train)
                
                # Calculate predictions for test set
                test_predictions = self.h_theta(self._x_test)
                correctly_classified_test = np.sum((test_predictions >= 0.5) == self._y_test)
                percent_correct_test = (correctly_classified_test / len(self._y_test)) * 100
                classified_correctly_test_list.append(percent_correct_test)

            # if cost < self.epsilon:
            #     break
            
            k += 1
        
        if metricMatrix:
            return train_metrics, test_metrics
        
        # print("Cost history: ", cost_history)
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################       
        return classified_correctly_train_list, classified_correctly_test_list, cost_history # Delete cost_history


    def train_binary_classification_sgd_cross_entropy(self, iterations, metricMatrix = False):
        """
        Performs a number of iterations of stochastic gradient descent (SGD) using cross-entropy loss.
        Returns a list with the percentage of instances classified correctly in the training and test sets.
        """
        classified_correctly_train_list_sgd = []
        classified_correctly_test_list_sgd = []

        ##############################################################################
        # TODO: Implement stochastic gradient descent algorithm on minimizing Cross  #
        # Entropy cost.                                                              #
        # You need to obtain partial derivatives before coding up your               # 
        # solutions. This method returns correctly classified train and test samples # 
        # computed at each iteration (Or each N iterations.)                         #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        # -1/N sum (y log(h_theta)) + (1 - y) log(1 - h_theta))
        k = 0
        weight_history = []
        cost_history = []
        train_metrics = []
        test_metrics = []

        # Stochastic Gradient Descent
        while k < iterations:
            X_copy = self._x_train.copy()
            
            while not X_copy.shape[1] == 0:
                idx = np.random.randint(0, X_copy.shape[1])
                x = X_copy[:, idx].reshape(-1, 1)
                X_copy = np.delete(X_copy, idx, axis=1)
                
                # Debug
                # print("X_copy shape: ", X_copy.shape)
                # print("X shape: ", x.shape)
                # print("W shape: ", self._W.shape)
                # print("Y shape: ", self._Y.shape)
                
                # Calculate the gradient
                grad_w = float(self.h_theta(x) - self._Y[:, idx]) * x.T / self._m
                
                grad_b = float(self.h_theta(x) - self._Y[:, idx]) / self._m
            
                # print("grad_w: ", grad_w)
                # print("grad_b: ", grad_b)
            
                self._W -= self._alpha * grad_w
                self._B -= self._alpha * grad_b 
                
            weight_history.append(self._W)

            cost = - np.sum(self._y_train * np.log(self.h_theta(self._x_train)) + (1 - self._y_train) * np.log(1 - self.h_theta(self._x_train))) / self._m
            # print("Cost: ", cost)
            
            cost_history.append(cost)
                
            if metricMatrix:
                # Calculate Metrics
                train_predictions = self.h_theta(self._x_train) >= .5
                test_predictions = self.h_theta(self._x_test) >= .5
                
                true_pos_train, true_neg_train, false_pos_train, false_neg_train = self.confusionMatrix(self._y_train, train_predictions)
                true_pos_test, true_neg_test, false_pos_test, false_neg_test = self.confusionMatrix(self._y_test, test_predictions)
                
                # print("--------------------")
                # print("True Pos Train: ", true_pos_train)
                # print("True Neg Train: ", true_neg_train)
                # print("False Pos Train: ", false_pos_train)
                # print("False Neg Train: ", false_neg_train)
                # print("--------------------")
                # print("True Pos Test: ", true_pos_test)
                # print("True Neg Test: ", true_neg_test)
                # print("False Pos Test: ", false_pos_test)
                # print("False Neg Test: ", false_neg_test)
                # print("--------------------")
                                
                # Calculate Metrics for Stochastic Gradient Descent (SGD) with Mean Squared Error (MSE)
                train_accuracy = (true_pos_train + true_neg_train) / (true_pos_train + true_neg_train + false_pos_train + false_neg_train)
                test_accuracy = (true_pos_test + true_neg_test) / (true_pos_test + true_neg_test + false_pos_test + false_neg_test)

                train_precision = true_pos_train / (true_pos_train + false_pos_train)
                test_precision = true_pos_test / (true_pos_test + false_pos_test)

                train_recall = true_pos_train / (true_pos_train + false_neg_train)
                test_recall = true_pos_test / (true_pos_test + false_neg_test)

                train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall)
                test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

                train_metrics.append([train_accuracy, train_precision, train_recall, train_f1]) # Hold as 2D array
                test_metrics.append([test_accuracy, test_precision, test_recall, test_f1])
                
            else:
                # Calculate predictions for training set
                train_predictions = self.h_theta(self._x_train)
                correctly_classified_train = np.sum((train_predictions >= 0.5) == self._y_train)
                percent_correct_train = (correctly_classified_train / len(self._y_train)) * 100
                # print("Percent correct train: ", percent_correct_train)
                classified_correctly_train_list_sgd.append(percent_correct_train)
                
                # Calculate predictions for test set
                test_predictions = self.h_theta(self._x_test)
                correctly_classified_test = np.sum((test_predictions >= 0.5) == self._y_test)
                percent_correct_test = (correctly_classified_test / len(self._y_test)) * 100
                classified_correctly_test_list_sgd.append(percent_correct_test)

            # if cost < self.epsilon:
            #     break
            
            k += 1
        
        if metricMatrix:
            return train_metrics, test_metrics
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ############################################################################## 

        return classified_correctly_train_list_sgd, classified_correctly_test_list_sgd, cost_history # Delete cost_history

