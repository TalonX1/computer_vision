





















from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0


    for i in range(num_train):
        scores = X[i].dot(W)                      # 根据权重计算的得分
        correct_class_score = scores[y[i]]        # 正确分类的得分

        for j in range(num_classes):
            if j == y[i]:            # 标签相同，计算损失
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1   损失函数
            if margin > 0:
                loss += margin

                # dw 维度(D,C) ,X的维度(N,D),dw的一列减去X的一行
                # dW[:, y[i]] += -X[i, :]     # 真实类的对应权重加上负的某一行的x值
                # dW[:, j] += X[i, :]         # 将标签相等的那一列权重置为0，结果就是，只有标签正确的那一输出为0，其他都为负数

                dW[:, j] += X[i].T  # 真实类的对应权重加上负的某一行的x值
                dW[:, y[i]] += -X[i].T  # 将标签相等的那一列权重


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    # loss += reg * np.sum(W * W)
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1)  # (N, 1)       # scores矩阵中样本分类正确的得分
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[range(num_train), list(y)] = 0
    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # loss /= num_train
    # dW /= num_train

    # Add regularization to the loss.
    # loss += reg * np.sum(W * W)

    coe_mat = np.zeros((num_train, num_classes))
    coe_mat[margins > 0] = 1
    coe_mat[range(num_train), list(y)] = 0
    coe_mat[range(num_train), list(y)] = -np.sum(coe_mat, axis=1)

    dW = (X.T).dot(coe_mat)
    dW = dW / num_train + reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
