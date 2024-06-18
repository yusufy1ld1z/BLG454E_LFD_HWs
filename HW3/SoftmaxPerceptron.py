import numpy as np
from tqdm import tqdm

np.random.seed(42)  # Set random seed if needed
class SoftmaxPerceptron:
    def __init__(self, weights, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights_ = weights
        
    def F(self, x):
        # print("[Function F] x not 0 shape: ", x[x != 0][10:15])
        # print("[Function F] x slice shape: ", x[10:15].shape)
        
        # print("[Function F] weights shape: ", self.weights[10:15]) # weights and weights_ are different
        return x.T @ self.weights + self.bias
     
    def sigmoid(self, Z):
        """
        Computes the sigmoid value for all values in vector Z.
        """                                  #
        
        sigmoid_val = 1 / (1 + np.exp(-Z))

        return sigmoid_val

    def h_theta(self, X):
        """
        Computes the value of the hypothesis according to the logistic regression rule.
        """                                #
        
        h_theta = self.sigmoid(X.T @ self.weights + self.bias) # check the shapes !!!
        
        return h_theta 
     
    @staticmethod
    def confusionMatrix(y_true, y_pred):
        # Create a confusion matrix
        true_pos = np.sum((y_pred == 1) & (y_true == 1)) # -1, 1 for y_true
        true_neg = np.sum((y_pred == -1) & (y_true == -1))
        false_pos = np.sum((y_pred == 1) & (y_true == -1))
        false_neg = np.sum((y_pred == -1) & (y_true == 1))
        
        return true_pos, true_neg, false_pos, false_neg
        
    def fit_sgd(self, X, y):
        np.random.seed(42)

        num_features, num_samples = X.shape  # Get the shape of the input data

        # Initialize weights
        self.weights = self.weights_[1:].reshape(-1, 1)
        self.bias = self.weights_[0]
        
        accuracy_history = []  # Initialize loss history

        ##############################################################################
        # TODO: Implement stochastic gradient descent (SGD) to train the logistic    #
        # regression model.                                                          #
        ##############################################################################
        # This function implements stochastic gradient descent (SGD) to train a      #
        # perceptron model. It iterates over the training samples for a              #
        # specified number of iterations (given by max_iter), updating the model     #
        # parameters (weights and bias) after processing each sample. SGD is a       #
        # variant of gradient descent that updates the parameters based on the       #
        # gradient of the loss function computed with respect to each individual     #
        # sample. This makes it computationally more efficient than batch gradient   #
        # descent, particularly for large datasets. The function computes the        #
        # logistic loss and its gradient for each sample, and updates the weights    #
        # and bias accordingly. The accuracy on the training set is computed after   #
        # each iteration and stored in the accuracy_history list. The function       #
        # returns this accuracy history to monitor the training progress.            #
        ##############################################################################

        # 1/N sum log (1 + e^(-y * (xT.w + b))) 
        
        k = 0
        
        # weight_history = []
        cost_history = []
        train_metrics = []
        test_metrics = []
        metricMatrix = True

        # Stochastic Gradient Descent
        print("[fit_sgd] Learning Started")
        
        while k < self.max_iter:
            print("[fit_sgd] Epoch: ", k)
            
            X_copy = X.copy()
            # with tqdm(total=num_samples, desc="Processing Samples") as pbar:  # tqdm progress bar # commented because it damages the visualization
            while not X_copy.shape[1] == 0:
                # print("[fit_sgd] Iteration: ", num_samples - X_copy.shape[1])
                
                idx = np.random.randint(0, X_copy.shape[1])
                x = X_copy[:, idx].reshape(-1, 1)
                X_copy = np.delete(X_copy, idx, axis=1)
                
                # Debug
                
                # Calculate the gradient
                # print("[fit_sgd] y: ", y.shape)
                # print("[fit_sgd] y[idx]: ", y[idx])
                
                # print("[fit_sgd] X: ", X.shape)
                # print("[fit_sgd] F(x): ", self.F(X).shape)
                # print((y.reshape(-1, 1) / (np.exp(y.reshape(-1, 1) * self.F(X)) + 1)).shape)
                
                # grad_w = - X @ (y.reshape(-1, 1) / (np.exp(y.reshape(-1, 1) * self.F(X)) + 1)) / num_samples 
                
                # grad_b = - np.sum(y.reshape(-1, 1) / (np.exp(y.reshape(-1, 1) * self.F(X)) + 1)) / num_samples 
                
                grad_w = - y[idx] / float(np.exp(y[idx] * self.F(x)) + 1) / num_samples * x 
                
                grad_b = - y[idx] / float(np.exp(y[idx] * self.F(x)) + 1) / num_samples
            
                # print("grad_w: ", grad_w[[10,15,20,25,30,35,40]]) # hep 0 veriyor
                # print("grad_b: ", grad_b)
            
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b 
                
                # pbar.update(1) # TQDM
                
            # weight_history.append(self._W)

            # print("[fit_sgd] F(X): ", self.F(X).shape)
            # print("[fit_sgd] y: ", y.shape)
            
            # print((self.F(X).T @ y.reshape(-1, 1)))
            # print(np.exp(-y.reshape(-1,1) * self.F(X)))
            
            cost = np.sum(np.log(1 + np.exp(-y.reshape(-1,1) * self.F(X)))) / num_samples # DOGRU OLAN BU
            
            print("[fit_sgd] Cost: ", cost)
            
            cost_history.append(cost)
                
            if metricMatrix:
                pass
                # Calculate Metrics
                train_predictions = self.predict(X)
                # test_predictions = self.h_theta(self._x_test) >= .5
                
                true_pos_train, true_neg_train, false_pos_train, false_neg_train = self.confusionMatrix(y, train_predictions)
                # true_pos_test, true_neg_test, false_pos_test, false_neg_test = self.confusionMatrix(self._y_test, test_predictions)
                
                # Calculate Metrics for Stochastic Gradient Descent (SGD) with Mean Squared Error (MSE)
                train_accuracy = (true_pos_train + true_neg_train) / (true_pos_train + true_neg_train + false_pos_train + false_neg_train)
                # test_accuracy = (true_pos_test + true_neg_test) / (true_pos_test + true_neg_test + false_pos_test + false_neg_test)

                train_precision = true_pos_train / (true_pos_train + false_pos_train)
                # test_precision = true_pos_test / (true_pos_test + false_pos_test)

                train_recall = true_pos_train / (true_pos_train + false_neg_train)
                # test_recall = true_pos_test / (true_pos_test + false_neg_test)

                train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall)
                # test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

                train_metrics.append([train_accuracy, train_precision, train_recall, train_f1]) # Hold as 2D array
                # test_metrics.append([test_accuracy, test_precision, test_recall, test_f1])
                
            else:
                # Calculate predictions for training set
                train_predictions = self.predict(X)
                # print(train_predictions[50:60])
                correctly_classified_train = np.sum(train_predictions == y) # np.sign function for -1, 1
                # print("[fit_sgd] Correctly classified train: ", correctly_classified_train)
                percent_correct_train = (correctly_classified_train / len(y)) * 100
                # print("Percent correct train: ", percent_correct_train)
                accuracy_history.append(percent_correct_train)
                    

            # if cost < self.epsilon:
            #     break
            
            k += 1
            
        if metricMatrix:
            return train_metrics, test_metrics

        print("[fit_sgd] Learning Finished")
        return accuracy_history

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")

        predictions = np.sign(np.dot(X.T, self.weights).flatten() + self.bias)
        return predictions
       
        
    def fit_gd_regularized(self, X, y, regularization_strength):
        np.random.seed(42)
        num_features, num_samples = X.shape  # Get the shape of the input data
        X_with_bias = X
        self.weights = self.weights_[1:].reshape(-1, 1)
        self.bias = self.weights_[0]

        accuracy_history = []  # Initialize empty list to store accuracy history
        ##############################################################################
        # TODO: Implement regularized gradient descent (GD) to train the softmax      #
        # perceptron model. The regularization_strength parameter controls the        #
        # strength of regularization applied to the model. Regularization helps      #
        # prevent overfitting by penalizing large parameter values. The function      #
        # iterates over the training samples for a specified number of iterations    #
        # (given by max_iter), updating the model parameters (weights and bias)       #
        # after processing each sample. The loss function used is the cross-entropy  #
        # loss with L2 regularization. It computes the loss and its gradient for     #
        # each sample, including a regularization term to penalize large weights.    #
        # The weights excluding the bias term are regularized using L2 regularization.#
        # The accuracy on the training set is computed after each iteration and      #
        # stored in the accuracy_history list. The function returns this accuracy    #
        # history to monitor the training progress.                                  #
        ##############################################################################

        # 1/N sum log (1 + e^(-y * (xT.w + b))) + lambda * w^2 
        
        k = 0
        
        # weight_history = []
        cost_history = []
        train_metrics = []
        test_metrics = []
        metricMatrix = False

        # Stochastic Gradient Descent
        print("[fit_gd_regularized] Learning Started")
        X_copy = X.copy()
        y_copy = y.copy().reshape(-1, 1)
        
        with tqdm(total=num_samples, desc="Processing Samples") as pbar:  # tqdm progress bar    
            while k < self.max_iter:
                
                # while not X_copy.shape[1] == 0:
                #     idx = np.random.randint(0, X_copy.shape[1])
                #     x = X_copy[:, idx].reshape(-1, 1)
                #     X_copy = np.delete(X_copy, idx, axis=1)
                    
                    # Debug
                    # print("X_copy shape: ", X_copy.shape)
                    # print("X shape: ", x.shape)
                    # print("W shape: ", self._W.shape)
                    # print("Y shape: ", self._Y.shape)
                    
                # print("[fit_gd_regularized] y: ", y.shape)
                # print("[fit_gd_regularized] weights: ", self.weights.shape)
                
                # print("[fit_gd_regularized] X: ", X_copy.shape)
                # print("[fit_gd_regularized] F(x): ", self.F(X_copy).shape)
                
                # Calculate the gradient
                grad_w = - X_copy @ (y_copy / (np.exp(y_copy * self.F(X_copy)) + 1)) / num_samples + regularization_strength * self.weights / num_samples
                # grad_w = float(self.h_theta(x) - y[idx]) / num_samples * x + regularization_strength * self.weights / num_samples
                
                grad_b = - np.sum(y_copy / (np.exp(y_copy * self.F(X_copy)) + 1)) / num_samples
                # grad_b = float(self.h_theta(x) - y[idx]) / num_samples
                
                # print("grad_w: ", grad_w)
                # print("grad_b: ", grad_b)
            
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b 
                    
                # weight_history.append(self._W)

                # cost = np.sum(np.log(1 + np.exp(-y.reshape(-1,1) * self.F(X_copy)))) / num_samples + regularization_strength * np.sum(self.weights ** 2) / num_samples
                
                # cost_history.append(cost)
                    
                if metricMatrix:
                    pass
                    # Calculate Metrics
                    # train_predictions = self.h_theta(self._x_train) >= .5
                    # test_predictions = self.h_theta(self._x_test) >= .5
                    
                    # true_pos_train, true_neg_train, false_pos_train, false_neg_train = self.confusionMatrix(self._y_train, train_predictions)
                    # true_pos_test, true_neg_test, false_pos_test, false_neg_test = self.confusionMatrix(self._y_test, test_predictions)
                    
                    # # Calculate Metrics for Stochastic Gradient Descent (SGD) with Mean Squared Error (MSE)
                    # train_accuracy = (true_pos_train + true_neg_train) / (true_pos_train + true_neg_train + false_pos_train + false_neg_train)
                    # test_accuracy = (true_pos_test + true_neg_test) / (true_pos_test + true_neg_test + false_pos_test + false_neg_test)

                    # train_precision = true_pos_train / (true_pos_train + false_pos_train)
                    # test_precision = true_pos_test / (true_pos_test + false_pos_test)

                    # train_recall = true_pos_train / (true_pos_train + false_neg_train)
                    # test_recall = true_pos_test / (true_pos_test + false_neg_test)

                    # train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall)
                    # test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

                    # train_metrics.append([train_accuracy, train_precision, train_recall, train_f1]) # Hold as 2D array
                    # test_metrics.append([test_accuracy, test_precision, test_recall, test_f1])
                    
                else:
                    # Calculate predictions for training set
                    train_predictions = self.predict(X)
                    correctly_classified_train = np.sum(np.sign(train_predictions) == y) # np.sign function for -1, 1
                    percent_correct_train = (correctly_classified_train / len(y)) * 100
                    # print("Percent correct train: ", percent_correct_train)
                    accuracy_history.append(percent_correct_train)
                    

                # if cost < self.epsilon:
                #     break
                
                pbar.update(1)
                k += 1
        
        if metricMatrix:
            return train_metrics, test_metrics
        
        print("[fit_gd_regularized] Learning Finished")

        return accuracy_history