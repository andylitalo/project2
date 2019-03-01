"""
List of methods for use in Project 2. Make sure to pull this when it is updated so your code
is compatible with the current version.
"""

import numpy as np
from prob2utils import train_model, get_RMSE
import pickle as pkl

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split


def method_1(Y_train, Y_test, reg=0.1, k=20, eta=0.03, eps=0.0001):
    """
    Runs method 1 over the given training parameters and returns a matrix of
    root-mean-squared error (RMSE).
    """
    # get number of users (M) and number of movies (N)
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    
    # train model
    U,V, ret = train_model(M, N, k, eta, reg, Y_train, eps=eps)

    # get error
    RMSE = get_RMSE(U, V, Y_test)
    
    return U, V, RMSE

def get_RMSE_off(model, testset):
    """
    Returns the RMSE of an off-the-shelf model over the test set.
    """
    # predict ratings for the testset
    predictions = model.test(testset)

    # Then compute RMSE
    RMSE = accuracy.rmse(predictions)
    
    return RMSE

def method_1_off(trainset, testset, k=20, eta=0.03, n_epochs=100, reg=0.1):
    """
    Uses off-the-shelf method without bias.
    """
    # set up model
    model = SVD(n_factors=k, n_epochs=n_epochs, biased=False, lr_all=eta, 
                     reg_all=reg, verbose=False)
    
    # train model
    model.fit(trainset)
    
    # get matrices
    U = model.pu
    V = model.qi
    
    # compute root-mean-square error
    RMSE = get_RMSE_off(model, testset)
    
    return U, V, RMSE

def method_2(Y_train, Y_test, reg=0.1, k=20, eta=0.03, eps=0.0001):
    """
    Method 2 using our own code. Adds bias term to factorization, but no 
    regularization. Returns matrix factors and root-mean-squared error.
    """
    # get number of users (M) and number of movies (N)
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    
    # train model with bias
    U,V,a,b, ret = train_model(M, N, k, eta, reg, Y_train, eps=eps, bias=True)
    
    # get root-mean-squared error over test data
    RMSE = get_RMSE(U, V, Y_test, bias=True, a=a, b=b)
    
    return U, V, a,b, RMSE

def method_23_off(trainset, testset, k=20, eta=0.03, n_epochs=100, 
                  reg=0.1, reg_bi=0, reg_bu=0):
    """
    Uses off-the-shelf method with bias but no regularization over the bias.
    """
    # set up model
    model = SVD(n_factors=k, n_epochs=n_epochs, biased=True, lr_all=eta, 
                     reg_all=reg, reg_bi=reg_bi, reg_bu=reg_bu, verbose=False)
    
    # train model
    model.fit(trainset)
    
    # get matrices
    U = model.pu
    V = model.qi
    ai = model.bu
    bj = model.bi
    
    # compute RMSE
    RMSE = get_RMSE_off(model, testset)
    
    return U, V, ai, bj, RMSE


    
if __name__=='__main__':
    # load training data
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)
    # also load using surprise library for compatibility with surprise packages
    data = Dataset.load_builtin('ml-100k')
    # sample random trainset and testset
    # test set is made of 10% of the ratings.
    trainset, testset = train_test_split(data, test_size=.10)
    
    # Method 1: from HW 5
    U1, V1, RMSE1 = method_1(Y_train, Y_test)
    # Method 1: off-the-shelf, no bias, no reg
    U1_off, V1_off, RMSE1_off = method_1_off(trainset, testset)
    # Method 2: add bias terms, no reg, our code
    U2, V2, a2, b2, RMSE2 = method_2(Y_train, Y_test)
    # Method 2: off the shelf
    U2_off, V2_off, a2_off, b2_off, RMSE2_off = method_23_off(trainset, testset)
    # Method 3: add bias and reg
    U3, V3, a3, b3, RMSE3 = method_23_off(trainset, testset, reg_bu=0.1, reg_bi=0.1)
    
    # compare performance
    print('Method 1: RMSE = %.4f' % RMSE1)
    print('Method 1 off the shelf: RMSE = %.4f' % RMSE1_off)
    print('Method 2: RMSE = %.4f' % RMSE2)
    print('Method 2 off the shelf: RMSE = %.4f' % RMSE2_off)
    print('Method 3: RMSE = %.4f' % RMSE3)
    
    # save results
    params = {}
    params['method 1'] = [np.transpose(U1), np.transpose(V1), RMSE1]
    params['method 2'] = [np.transpose(U2), np.transpose(V2), a2, b2, RMSE2]
    params['method 3'] = [np.transpose(U3), np.transpose(V3), a3, b3, RMSE3]
    
    with open('data/matrix_factorization.pkl', 'wb') as f:
        pkl.dump(params, f)
