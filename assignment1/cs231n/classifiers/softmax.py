import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  dim = W.shape[1]
  for i in xrange(num_train):
     scores = np.dot(X[i,:], W)
     softmax = np.exp(scores) / np.sum(np.exp(scores))
     loss -= np.log(softmax[y[i]])
     for j in xrange(dim):
        if y[i] != j:
          dW[:,j] += softmax[j] * X[i,:]
        else:
          dW[:,y[i]] -= (1 - softmax[y[i]]) * X[i,:]
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  softmax = np.exp(scores) / np.sum(np.exp(scores), axis = 1).reshape(-1, 1)
  correct_table = np.zeros(scores.shape)
  correct_table[range(num_train), y] = 1
  loss -= np.sum(np.log(np.sum(np.multiply(softmax, correct_table), axis = 1)))
  dW = np.dot(X.T, np.multiply(1 - correct_table, softmax)) - np.dot(X.T, np.multiply(correct_table, 1 - softmax))
  
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

