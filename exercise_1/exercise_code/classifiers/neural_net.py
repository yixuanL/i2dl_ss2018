"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape
        d, c = W2.shape
        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################
        
        X1 = np.dot(X, W1) + b1
        X1 = np.maximum(X1,0)
        scores = np.dot(X1, W2) + b2

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################
        # compute loss:
        maxScore = np.amax(scores)
        scores -= maxScore
        ex = np.exp(scores)
        ex += 1e-7
        sum_ex = np.sum(ex, axis=1)
        sum_ex = np.tile(sum_ex, (c,1)).T
        tmp_pred_y = ex/(sum_ex)

        equal_y = np.zeros_like(tmp_pred_y)
        equal_y = np.tile(y,(c,1)).T
        tmp_arr = np.arange(c)
        tmp_arr = np.tile(tmp_arr,(N,1))
        tmp_mul_arr = np.equal(tmp_arr, equal_y).astype(int)

        loss = -np.sum(np.log(tmp_pred_y)*tmp_mul_arr)
        # final results:
        loss /= N
        loss += 0.5*np.sum(W1*W1)*reg + 0.5*np.sum(W2*W2)*reg 
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################
        gradLoss = np.ones((N,c))*(tmp_mul_arr) - tmp_pred_y
        gradLoss /= -N
        dW2 = np.dot(X1.T, gradLoss) + reg*W2 
        db2 = np.sum(gradLoss, axis = 0)
        dX1 = np.dot(gradLoss,W2.T)
        dX1[X1 <= 0] = 0
        dW1 = np.dot(X.T, dX1) + reg*W1
        db1 = np.sum(dX1, axis = 0)
        grads['W1'] = dW1
        grads['W2'] = dW2 
        grads['b1'] = db1
        grads['b2'] = db2
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################

            idx = np.random.choice(num_train, batch_size)
            X_batch = X[idx,:]
            y_batch = y[idx,]
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################

            self.params['W1'] -= grads['W1'] * learning_rate
            self.params['W2'] -= grads['W2'] * learning_rate
            self.params['b1'] -= grads['b1'] * learning_rate
            self.params['b2'] -= grads['b2'] * learning_rate
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################


            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                # Decay learning rate
                learning_rate *= learning_rate_decay
            if verbose and it % 100 == 0:
                val_acc = (self.predict(X_val) == y_val).mean()
                print('iteration %d / %d: loss %f, acc: %f' % (it, num_iters, loss, val_acc))

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################

        y_pred = np.dot(X, self.params['W1']) + self.params['b1']
        y_pred = np.maximum(y_pred,0)
        y_pred = np.dot(y_pred, self.params['W2']) + self.params['b2']
        y_pred = np.argmax(y_pred, axis=1)
        
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred

def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val,input_dim= 32*32*3, hidden_dim = 620):
    best_net = None # store the best model into this 

    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above; these visualizations will have significant      #
    # qualitative differences from the ones we saw above for the poorly tuned  #
    # network.                                                                 #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################
    best_val = -1
    learning_rates = [5e-4]
    regularization_strengths = [1e-3]
    hidden_sizes = [hidden_dim]
    learning_rate_decays = [0.975]
    stds = [4e-3]
    
    for k in range(len(hidden_sizes)):
        for i in range(len(learning_rates)):
            for j in range(len(regularization_strengths)):
                for l in range(len(learning_rate_decays)):
                    for m in range(len(stds)):
                        nn = TwoLayerNet(input_dim, hidden_sizes[k], 10, std=stds[m])
                        loss_hist = nn.train(X_train, y_train, X_val, y_val, 
                                             learning_rate=learning_rates[i],
                                             learning_rate_decay=learning_rate_decays[l],
                                             reg=regularization_strengths[j], 
                                             num_iters=5000, verbose=True)
                        pred_y_train = nn.predict(X_train)
                        pred_y_val = nn.predict(X_val)
                        if(np.mean(pred_y_val == y_val) > best_val) :
                            best_val = np.mean(pred_y_val == y_val)
                            best_net = nn
                            print(best_val)


    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
