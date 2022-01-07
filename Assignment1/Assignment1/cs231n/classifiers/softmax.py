from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)              # 行向量
        scores_shift = np.max(scores)     # 将得分的最大值设置为偏移量
        scores -= scores_shift

        loss += np.log(np.sum(np.exp(scores)))-scores[y[i]]      # 当前得分行向量全部指数求和的值-正确标签对应的得分
        dW[:, y[i]] -= X[i]                                      # dW正确标签对应的列向量减去X行向量

        scores_sum = np.exp(scores).sum()
        for j in range(num_classes):
            dW[:, j] += np.exp(scores[j]) / scores_sum * X[i]

    loss = loss / num_train + 0.5 * reg * np.sum(W*W)
    dW = dW / num_train + reg * W




    # loss /= num_train
    # dW /= num_train

    pass



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = np.dot(X, W)
    # scores_shift = np.max(scores, axis=1)
    # print(scores_shift.shape)
    # scores -= scores_shift
    scores -= scores.max(axis=1).reshape(num_train, 1)
    scores_sum = np.exp(scores).sum(axis=1)

    loss = np.log(scores_sum).sum() - scores[range(num_train), y].sum()

    counts = np.exp(scores) / scores_sum.reshape(num_train, 1)
    counts[range(num_train),y] -= 1
    dW = np.dot(X.T, counts)

    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W
    pass


    return loss, dW
