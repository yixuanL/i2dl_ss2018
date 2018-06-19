"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    n = X.shape[0]
    d = W.shape[0]
    c = W.shape[1]
    
    # compute predicted y:
    tmp_pred_y = np.zeros((n,c))
    #pred_y = np.zeros_like(y)
    #equal_y = np.zeros_like(y)
    
    for i in range(n):
        for j in range(d):
            for k in range(c):
                tmp_pred_y[i][k] += X[i][j]*W[j][k]
                
    for p in range(n):
        ex = np.zeros(c)
        sum_ex = 0
        #max_index = -1
        for q in range(c):
            ex[q] = np.exp(tmp_pred_y[p][q])
            sum_ex += ex[q]
        for q in range(c):
            tmp_pred_y[p][q] = ex[q]/sum_ex
        #max_index = np.argmax(tmp_pred_y[p])
        #pred_y[p] = tmp_pred_y[p][max_index]
        
        #if(y[p] == max_index): equal_y[p] = 1
        #else: equal_y[p] = 0
        
    # compute loss:
        #loss += np.log( tmp_pred_y[p][max_index]) * equal_y[p] + (1-equal_y[p]) * np.log(1- tmp_pred_y[p][max_index])
        #loss += np.log( tmp_pred_y[p][max_index])
        loss += np.log(tmp_pred_y[p][y[p]])
        
    # compute gradient of W:
        for k in range(d):
            for l in range(c):
                
                if(y[p]==l):
                    dW[k][l] += X[p][k] * (1 - tmp_pred_y[p][l])
                else:
                    dW[k][l] -= X[p][k] * tmp_pred_y[p][l]
    
    # final results:
    loss /= -n
    loss += 0.5*np.sum(W*W)*reg
    dW /= -n
    dW += reg*W
    
        
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    n = X.shape[0]
    d = W.shape[0]
    c = W.shape[1]
    
    # compute predicted y:
    tmp_pred_y = np.zeros((n,c))
    tmp_pred_y += np.dot(X,W)
    #tmp_max = np.amax(tmp_pred_y)
    #tmp_pred_y -= tmp_max
    
    # compute loss:
    ex = np.zeros_like(tmp_pred_y)
    sum_ex = np.zeros(n)
    
    ex = np.exp(tmp_pred_y)
    sum_ex = np.sum(ex, axis=1)
    sum_ex = np.tile(sum_ex, (c,1)).T
    tmp_pred_y = ex/sum_ex
    
    equal_y = np.zeros_like(tmp_pred_y)
    equal_y = np.tile(y,(c,1)).T
    tmp_arr = np.arange(c)
    tmp_arr = np.tile(tmp_arr,(n,1))
    tmp_mul_arr = np.equal(tmp_arr, equal_y).astype(int)
    
    loss = np.sum(np.log(tmp_pred_y)*tmp_mul_arr)
    
    #compute gradient of W:
    dW = np.dot(X.T, (np.ones((n,c))*(tmp_mul_arr) - tmp_pred_y))
    
    # final results:
    loss /= -n
    loss += 0.5*np.sum(W*W)*reg
    dW /= -n
    dW += reg*W
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [2e-6]
    regularization_strengths = [1e2]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    for i in range(len(learning_rates)):
        for j in range(len(regularization_strengths)):
            softmax = SoftmaxClassifier()
            loss_hist = softmax.train(X_train, y_train, learning_rate=learning_rates[i], reg=regularization_strengths[j],
                          num_iters=30000, verbose=True)
            pred_y_train = softmax.predict(X_train)
            pred_y_val = softmax.predict(X_val)
            if(np.mean(pred_y_val == y_val) > best_val) :
                best_val = np.mean(pred_y_val == y_val)
                best_softmax = softmax
            results[(learning_rates[i],regularization_strengths[j])] = (np.mean(pred_y_train == y_train),np.mean(pred_y_val == y_val))
            
            
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
